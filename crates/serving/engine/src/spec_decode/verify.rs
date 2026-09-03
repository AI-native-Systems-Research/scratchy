// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Greedy rejection sampling for spec-decode verify steps.
//!
//! Pure host fn — no GPU dependency. Takes target's per-row argmax over
//! the verify-batch logits and the drafts that batch was built from, and
//! returns the accepted prefix plus the recovered/bonus token.
//!
//! Matches Python vLLM's `_rejection_sample_kernel` for greedy decoding.

/// Result of greedy rejection sampling for a single request.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectionResult {
    /// Accepted token IDs (target argmax values). Length is 1..=num_drafts+1.
    /// On full acceptance: K target-verified tokens + 1 bonus token.
    /// On partial acceptance: M target-verified tokens + 1 recovered token.
    /// On first rejection: 1 recovered token.
    pub accepted_tokens: Vec<u32>,
    /// Number of draft tokens that were accepted (0..=num_drafts).
    pub num_accepted_drafts: usize,
}

/// Perform greedy rejection sampling for one request.
///
/// Given `target_ids` (argmax of the model's logits at each position) and
/// `draft_token_ids` (proposed draft tokens), accept the longest prefix of
/// matching drafts and return the accepted tokens.
///
/// Layout:
/// - `target_ids[0]` = argmax at the real token position (verifies draft[0])
/// - `target_ids[i]` = argmax at draft[i-1] position (verifies draft[i])
/// - `target_ids[K]` = bonus token (only used if all K drafts accepted)
pub fn greedy_rejection_sample(target_ids: &[u32], draft_token_ids: &[u32]) -> RejectionResult {
    if draft_token_ids.is_empty() {
        // Normal request: no drafts, just 1 token.
        return RejectionResult {
            accepted_tokens: vec![target_ids[0]],
            num_accepted_drafts: 0,
        };
    }

    let mut accepted = Vec::with_capacity(draft_token_ids.len() + 1);

    for i in 0..draft_token_ids.len() {
        let target = target_ids[i];
        accepted.push(target);
        if target != draft_token_ids[i] {
            // Mismatch: target is the recovered token. Stop.
            return RejectionResult {
                num_accepted_drafts: i,
                accepted_tokens: accepted,
            };
        }
    }

    // All drafts accepted — append bonus token from the last logit position.
    let bonus = target_ids[draft_token_ids.len()];
    accepted.push(bonus);
    RejectionResult {
        num_accepted_drafts: draft_token_ids.len(),
        accepted_tokens: accepted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_drafts() {
        let r = greedy_rejection_sample(&[7], &[]);
        assert_eq!(r.accepted_tokens, vec![7]);
        assert_eq!(r.num_accepted_drafts, 0);
    }

    #[test]
    fn first_mismatch() {
        let r = greedy_rejection_sample(&[5, 6, 7], &[1, 2]);
        assert_eq!(r.accepted_tokens, vec![5]);
        assert_eq!(r.num_accepted_drafts, 0);
    }

    #[test]
    fn partial_accept() {
        let r = greedy_rejection_sample(&[1, 9, 8], &[1, 2]);
        assert_eq!(r.accepted_tokens, vec![1, 9]);
        assert_eq!(r.num_accepted_drafts, 1);
    }

    #[test]
    fn full_accept_with_bonus() {
        let r = greedy_rejection_sample(&[1, 2, 3, 99], &[1, 2, 3]);
        assert_eq!(r.accepted_tokens, vec![1, 2, 3, 99]);
        assert_eq!(r.num_accepted_drafts, 3);
    }
}
