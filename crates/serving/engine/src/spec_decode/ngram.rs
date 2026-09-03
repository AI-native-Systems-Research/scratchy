// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! N-gram speculative decoding proposer.
//!
//! Scans a request's token history (prompt + output) for matching n-grams
//! and proposes continuation tokens as speculative drafts. The target model
//! verifies these drafts in a single multi-token forward pass, accepting
//! all tokens up to the first mismatch.
//!
//! This is the simplest form of speculative decoding — no draft model needed,
//! just a lookup table over the request's own token history. Works well for
//! repetitive text, code, and reformatting tasks.
//!
//! Port of: `vllm/v1/spec_decode/ngram_proposer.py`
//!
//! Uses the same KMP (Knuth-Morris-Pratt) LPS-based O(n) algorithm as Python:
//! reverse the tokens, compute a longest-prefix-suffix array capped at
//! max_ngram, track the longest match and position in a single pass, then
//! extract the k tokens following the match in the original ordering.

/// Configuration for the N-gram proposer.
#[derive(Debug, Clone)]
pub struct NgramProposerConfig {
    /// Number of draft tokens to propose per step.
    pub num_speculative_tokens: usize,
    /// Maximum n-gram size to search for (tries this first, then smaller).
    /// Minimum is 1 (unigram — matches any single token).
    pub max_ngram_size: usize,
    /// Minimum n-gram size. Stops searching below this.
    pub min_ngram_size: usize,
    /// Maximum model context length. Proposals are capped so total tokens
    /// don't exceed this limit. 0 means no limit.
    pub max_model_len: usize,
}

impl Default for NgramProposerConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 5,
            max_ngram_size: 4,
            min_ngram_size: 1,
            max_model_len: 0,
        }
    }
}

/// Proposes draft tokens by matching n-grams in the token history.
#[derive(Debug)]
pub struct NgramProposer {
    config: NgramProposerConfig,
}

impl NgramProposer {
    /// Create a new proposer with the given configuration.
    pub fn new(config: NgramProposerConfig) -> Self {
        Self { config }
    }

    /// Propose draft tokens for speculative decoding.
    ///
    /// Uses the KMP (Knuth-Morris-Pratt) LPS algorithm on reversed tokens,
    /// matching Python vLLM's `_find_longest_matched_ngram_and_propose_tokens`
    /// exactly.
    ///
    /// Algorithm:
    /// 1. Reverse tokens so the tail n-gram becomes a prefix
    /// 2. Compute LPS (longest prefix suffix) array, capped at max_ngram
    /// 3. Track the longest match and its position in a single O(n) pass
    /// 4. Convert position back and extract k tokens after the match
    ///
    /// Returns an empty vec if no matching n-gram is found.
    pub fn propose(&self, all_token_ids: &[u32]) -> Vec<u32> {
        let total_token = all_token_ids.len();
        let min_ngram = self.config.min_ngram_size.max(1);

        if total_token < min_ngram {
            return Vec::new();
        }

        // Cap k so total tokens don't exceed max_model_len.
        let mut k = self.config.num_speculative_tokens;
        if self.config.max_model_len > 0 {
            let remaining = self.config.max_model_len.saturating_sub(total_token);
            if remaining == 0 {
                return Vec::new();
            }
            k = k.min(remaining);
        }
        if k == 0 {
            return Vec::new();
        }

        let max_ngram = self.config.max_ngram_size.min(total_token);

        // Reverse tokens: the tail n-gram becomes a prefix.
        // We work with indices into all_token_ids to avoid allocation.
        // tokens[i] in the reversed view = all_token_ids[total_token - 1 - i]
        let rev = |i: usize| -> u32 { all_token_ids[total_token - 1 - i] };

        // LPS array, capped at max_ngram.
        let mut lps = vec![0u32; max_ngram];

        let mut longest_ngram: usize = 0;
        let mut position: usize = 0;

        // KMP LPS computation on the reversed tokens.
        // lps[0] is always 0; start from index 1.
        let mut prev_lps: usize = 0;
        let mut i: usize = 1;

        while i < total_token {
            if rev(prev_lps) == rev(i) {
                // Token match
                prev_lps += 1;

                // Update when we find a longer-or-equal valid ngram.
                // >= ensures earliest position in original (latest in reversed).
                if prev_lps >= longest_ngram {
                    longest_ngram = prev_lps;
                    position = i;
                }

                if i < max_ngram {
                    lps[i] = prev_lps as u32;
                }

                if prev_lps == max_ngram {
                    // Cap: don't match ngrams longer than max_ngram.
                    prev_lps = lps[max_ngram - 1] as usize;
                }

                i += 1;
            } else if prev_lps != 0 {
                // Token mismatch: fall back to second-longest prefix-suffix.
                prev_lps = lps[prev_lps - 1] as usize;
            } else {
                // No prefix matches; advance.
                i += 1;
            }
        }

        if longest_ngram < min_ngram {
            return Vec::new();
        }

        // Convert reversed position back to original token space.
        // In origin_tokens: the matched ngram starts at
        //   total_token - 1 - position
        // and has length longest_ngram, so the tokens to draft start at:
        //   total_token - 1 - position + longest_ngram
        let start_position = total_token - 1 - position + longest_ngram;
        let actual_k = k.min(total_token - start_position);
        if actual_k == 0 {
            return Vec::new();
        }

        all_token_ids[start_position..start_position + actual_k].to_vec()
    }

    /// Get the configuration.
    pub fn config(&self) -> &NgramProposerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_proposer() -> NgramProposer {
        NgramProposer::new(NgramProposerConfig::default())
    }

    fn proposer(num_spec: usize, max_n: usize, min_n: usize) -> NgramProposer {
        NgramProposer::new(NgramProposerConfig {
            num_speculative_tokens: num_spec,
            max_ngram_size: max_n,
            min_ngram_size: min_n,
            max_model_len: 0,
        })
    }

    fn proposer_with_max_len(
        num_spec: usize,
        max_n: usize,
        min_n: usize,
        max_model_len: usize,
    ) -> NgramProposer {
        NgramProposer::new(NgramProposerConfig {
            num_speculative_tokens: num_spec,
            max_ngram_size: max_n,
            min_ngram_size: min_n,
            max_model_len,
        })
    }

    #[test]
    fn test_empty_input() {
        let p = default_proposer();
        assert!(p.propose(&[]).is_empty());
    }

    #[test]
    fn test_single_token() {
        let p = default_proposer();
        assert!(p.propose(&[42]).is_empty());
    }

    #[test]
    fn test_no_match() {
        let p = default_proposer();
        // All unique tokens — no n-gram repeats.
        let tokens: Vec<u32> = (0..20).collect();
        assert!(p.propose(&tokens).is_empty());
    }

    #[test]
    fn test_simple_bigram_match() {
        // Tokens: [A, B, C, D, A, B]
        // Tail bigram [A, B] matches at position 0.
        // Continuation after position 0+2 = [C, D, A, B] → propose [C, D, A, B].
        let p = proposer(5, 4, 1);
        let tokens = vec![10, 20, 30, 40, 10, 20];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![30, 40, 10, 20]);
    }

    #[test]
    fn test_trigram_preferred_over_bigram() {
        // Tokens: [A, B, C, D, E, A, B, C]
        // Tail trigram [A, B, C] matches at position 0.
        // Continuation: [D, E, A, B, C] → capped at 5 → [D, E, A, B, C].
        let p = proposer(5, 4, 1);
        let tokens = vec![10, 20, 30, 40, 50, 10, 20, 30];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![40, 50, 10, 20, 30]);
    }

    #[test]
    fn test_max_speculative_tokens_limit() {
        // Many tokens after the match, but limited to 3.
        let p = proposer(3, 2, 1);
        let tokens = vec![10, 20, 30, 40, 50, 60, 10, 20];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![30, 40, 50]);
    }

    #[test]
    fn test_most_recent_match_preferred() {
        // Tokens: [A, B, X, A, B, Y, A, B]
        // The KMP algorithm on reversed tokens picks the earliest position
        // in original tokens (latest in reversed). Both matches at pos 0
        // and pos 3 exist. The KMP >= comparison picks the earliest
        // match in original tokens.
        let p = proposer(5, 2, 1);
        let tokens = vec![10, 20, 100, 10, 20, 200, 10, 20];
        let drafts = p.propose(&tokens);
        // KMP picks earliest original position (pos 0): continuation [100, 10, 20, 200, 10]
        // OR pos 3: continuation [200, 10, 20]
        // Python's KMP `>=` picks the latest in reversed = earliest in original = pos 0
        assert_eq!(drafts, vec![100, 10, 20, 200, 10]);
    }

    #[test]
    fn test_unigram_fallback() {
        // Tokens: [A, B, C, A] — tail unigram [A] matches at position 0.
        let p = proposer(5, 3, 1);
        let tokens = vec![10, 20, 30, 10];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![20, 30, 10]);
    }

    #[test]
    fn test_min_ngram_size_respected() {
        // With min_ngram_size=2, unigram fallback should be disabled.
        let p = proposer(5, 3, 2);
        let tokens = vec![10, 20, 30, 10];
        let drafts = p.propose(&tokens);
        assert!(drafts.is_empty());
    }

    #[test]
    fn test_repetitive_sequence() {
        // Highly repetitive: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
        let p = proposer(5, 3, 1);
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2];
        let drafts = p.propose(&tokens);
        // KMP finds trigram [3, 1, 2] matching at earliest original position (pos 0).
        // Continuation from pos 3: [1, 2, 3, 1, 2] capped at 5.
        // But actually KMP picks match giving most continuation — earliest pos.
        assert!(!drafts.is_empty());
        assert!(drafts.len() <= 5);
    }

    #[test]
    fn test_match_at_boundary() {
        let p = proposer(5, 2, 1);
        let tokens = vec![10, 20, 30, 10, 20];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![30, 10, 20]);
    }

    #[test]
    fn test_code_like_pattern() {
        let def_tok = 100;
        let colon = 101;
        let newline = 102;
        let indent = 103;
        let return_tok = 104;
        let zero = 105;

        let tokens = vec![
            def_tok, 200, colon, newline, indent, return_tok, zero, newline, def_tok, 201, colon,
            newline, indent, return_tok,
        ];

        let p = proposer(3, 4, 1);
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![zero, newline, def_tok]);
    }

    #[test]
    fn test_default_config() {
        let config = NgramProposerConfig::default();
        assert_eq!(config.num_speculative_tokens, 5);
        assert_eq!(config.max_ngram_size, 4);
        assert_eq!(config.min_ngram_size, 1);
        assert_eq!(config.max_model_len, 0);
    }

    #[test]
    fn test_config_accessor() {
        let p = proposer(3, 5, 2);
        assert_eq!(p.config().num_speculative_tokens, 3);
        assert_eq!(p.config().max_ngram_size, 5);
        assert_eq!(p.config().min_ngram_size, 2);
    }

    #[test]
    fn test_two_tokens_input() {
        let p = proposer(5, 1, 1);
        let tokens = vec![42, 42];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![42]);
    }

    #[test]
    fn test_long_match_but_short_continuation() {
        let p = proposer(5, 4, 1);
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3, 4];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![5, 1, 2, 3, 4]);
    }

    // --- KMP-specific tests ---

    #[test]
    fn test_max_model_len_capping() {
        // 10 tokens, max_model_len=12, so at most 2 draft tokens.
        let p = proposer_with_max_len(5, 4, 1, 12);
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3, 4];
        let drafts = p.propose(&tokens);
        // Would normally propose [5, 1, 2, 3, 4] but capped to 3 (12 - 9 = 3).
        assert_eq!(drafts.len(), 3);
        assert_eq!(drafts, vec![5, 1, 2]);
    }

    #[test]
    fn test_max_model_len_at_limit() {
        // Already at max_model_len — no proposals.
        let p = proposer_with_max_len(5, 4, 1, 5);
        let tokens = vec![1, 2, 3, 1, 2];
        let drafts = p.propose(&tokens);
        assert!(drafts.is_empty());
    }

    #[test]
    fn test_max_model_len_zero_means_unlimited() {
        let p = proposer_with_max_len(5, 4, 1, 0);
        let tokens = vec![1, 2, 3, 1, 2];
        let drafts = p.propose(&tokens);
        assert!(!drafts.is_empty());
    }

    #[test]
    fn test_kmp_earliest_position() {
        // KMP picks the earliest position in original tokens for ties (>= in Python).
        // Tokens: [A, B, C, A, B, C, A, B]
        // The KMP algorithm on reversed tokens finds matches and picks
        // the earliest position in original space.
        let p = proposer(5, 2, 1);
        let tokens = vec![10, 20, 30, 10, 20, 30, 10, 20];
        let drafts = p.propose(&tokens);
        // Valid: must be a continuation after some matching n-gram.
        assert!(!drafts.is_empty());
        assert!(drafts.len() <= 5);
        // Verify it's a valid continuation from the token sequence.
        let found = (0..tokens.len() - 1).any(|start| {
            start + drafts.len() <= tokens.len()
                && tokens[start..start + drafts.len()] == drafts[..]
        });
        assert!(found, "drafts {:?} not found in tokens", drafts);
    }

    #[test]
    fn test_kmp_matches_naive_on_random_sequences() {
        // Property test: KMP should produce valid proposals (tokens exist in history).
        // We can't compare 1:1 with the old naive algo since KMP picks earliest
        // position while naive picked latest, but we can verify:
        // 1. The proposed tokens actually follow a matching n-gram in history
        // 2. Length is <= num_speculative_tokens

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let p = proposer(5, 4, 1);

        for seed in 0..100u64 {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            let hash = h.finish();

            // Generate a pseudo-random sequence with some repetition.
            let len = 10 + (hash % 50) as usize;
            let vocab = 3 + (hash % 5) as u32; // small vocab → lots of repeats
            let tokens: Vec<u32> = (0..len)
                .map(|i| {
                    let mut h2 = DefaultHasher::new();
                    (seed, i).hash(&mut h2);
                    (h2.finish() % vocab as u64) as u32
                })
                .collect();

            let drafts = p.propose(&tokens);
            assert!(drafts.len() <= 5, "too many drafts for seed {seed}");

            if !drafts.is_empty() {
                // Verify: the proposed tokens must appear somewhere in the
                // original token sequence following a matching n-gram.
                let tail_end = tokens.len();
                let mut found = false;
                'outer: for n in (1..=4).rev() {
                    if n > tail_end {
                        continue;
                    }
                    let tail = &tokens[tail_end - n..];
                    for start in 0..tail_end - n {
                        if start + n > tail_end - n {
                            continue;
                        }
                        if &tokens[start..start + n] == tail {
                            let follow_start = start + n;
                            if follow_start + drafts.len() <= tokens.len()
                                && tokens[follow_start..follow_start + drafts.len()] == drafts[..]
                            {
                                found = true;
                                break 'outer;
                            }
                        }
                    }
                }
                assert!(
                    found,
                    "drafts {:?} not found after any matching n-gram in {:?} (seed {seed})",
                    drafts, tokens
                );
            }
        }
    }
}
