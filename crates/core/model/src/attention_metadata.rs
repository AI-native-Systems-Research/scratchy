// SPDX-License-Identifier: Apache-2.0
//! Attention metadata for cross-request batched forward passes.
//!
//! `AttentionMetadata` describes how tokens from multiple requests are
//! packed into a single flat `[total_tokens, ...]` tensor. The per-request
//! slicing information lets attention layers split Q/K/V back to per-request
//! tensors while all other ops (embedding, projections, norms, MLP, lm_head)
//! run on the full batched tensor.

/// Metadata describing a batch of requests packed into flat tensors.
///
/// All token IDs and positions are concatenated in request order:
/// `[req_0_tokens..., req_1_tokens..., ...]`. The `query_start_loc` offsets
/// let attention layers narrow to per-request slices.
#[derive(Debug, Clone)]
pub struct AttentionMetadata {
    /// Number of requests in this batch.
    pub num_reqs: usize,
    /// Total number of tokens across all requests.
    pub total_tokens: usize,
    /// Cumulative token offsets, length `num_reqs + 1`.
    /// `query_start_loc[i]` is the start of request `i`'s tokens in the flat tensor.
    /// `query_start_loc[num_reqs]` == `total_tokens`.
    pub query_start_loc: Vec<usize>,
    /// Tokens scheduled for each request (prefill: prompt len, decode: 1).
    pub q_lens: Vec<usize>,
    /// Total context length per request (cached + new tokens).
    pub seq_lens: Vec<usize>,
    /// Per-request paged block IDs for the FULL/global KV-cache group (group 0).
    pub block_ids: Vec<Vec<usize>>,
    /// Per-SLIDING-GROUP, per-request paged block IDs (vLLM group-shared
    /// layout): `sliding_groups[s][req][logical]`. gemma4 has N sliding groups
    /// (group `s` here = KV-cache group `s + 1`); group 0 = full uses
    /// [`Self::block_ids`]. Empty on non-SWA models (every layer → group 0).
    pub sliding_groups: Vec<Vec<Vec<usize>>>,
    /// Tokens already in block pool per request.
    pub tokens_before: Vec<usize>,
    /// Whether each request is prefill vs decode.
    pub is_prefill: Vec<bool>,
    /// Request IDs in batch order.
    pub req_ids: Vec<String>,
}

impl AttentionMetadata {
    /// Create a new `AttentionMetadata`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_reqs: usize,
        total_tokens: usize,
        query_start_loc: Vec<usize>,
        q_lens: Vec<usize>,
        seq_lens: Vec<usize>,
        block_ids: Vec<Vec<usize>>,
        tokens_before: Vec<usize>,
        is_prefill: Vec<bool>,
        req_ids: Vec<String>,
    ) -> Self {
        Self {
            num_reqs,
            total_tokens,
            query_start_loc,
            q_lens,
            seq_lens,
            block_ids,
            sliding_groups: Vec::new(),
            tokens_before,
            is_prefill,
            req_ids,
        }
    }

    /// Attach the sliding KV-cache groups' per-request block tables (gemma4
    /// SWA): `sliding[s][req][block]`. Empty keeps the single-group behavior.
    pub fn with_sliding_groups(mut self, sliding: Vec<Vec<Vec<usize>>>) -> Self {
        self.sliding_groups = sliding;
        self
    }

    /// Returns `(start_offset, q_len)` for request `req_idx` in the flat tensor.
    pub fn request_slice(&self, req_idx: usize) -> (usize, usize) {
        let start = self.query_start_loc[req_idx];
        let q_len = self.q_lens[req_idx];
        (start, q_len)
    }

    /// True if every request in the batch is a decode (q_len == 1).
    pub fn is_all_decode(&self) -> bool {
        self.is_prefill.iter().all(|&p| !p)
    }

    /// Indices of the last token per request in the flat `[total_tokens]` tensor.
    ///
    /// For decode requests (q_len=1) this is the token itself. For prefill
    /// requests this is the last prompt token. Used to gather hidden states
    /// before the LM head so we only compute `[num_reqs, vocab]` logits
    /// instead of `[total_tokens, vocab]`.
    pub fn sample_indices(&self) -> Vec<u32> {
        self.query_start_loc[..self.num_reqs]
            .iter()
            .zip(&self.q_lens)
            .map(|(&start, &qlen)| (start + qlen - 1) as u32)
            .collect()
    }

    /// Create metadata for a padded all-decode batch (CUDA graph capture/replay).
    ///
    /// All requests are decode (q_len=1). Padded slots (`actual_bs..padded_bs`)
    /// use dummy `seq_lens=1`, a single dummy block ID (block 0), and
    /// `tokens_before=0`.
    pub fn padded_decode(
        padded_bs: usize,
        actual_bs: usize,
        seq_lens: &[usize],
        block_ids: &[Vec<usize>],
        tokens_before: &[usize],
    ) -> Self {
        debug_assert!(actual_bs <= padded_bs);
        debug_assert_eq!(seq_lens.len(), actual_bs);
        debug_assert_eq!(block_ids.len(), actual_bs);
        debug_assert_eq!(tokens_before.len(), actual_bs);

        let mut full_seq_lens = seq_lens.to_vec();
        let mut full_block_ids = block_ids.to_vec();
        let mut full_tokens_before = tokens_before.to_vec();
        let mut full_is_prefill = vec![false; actual_bs];
        let mut full_req_ids: Vec<String> =
            (0..actual_bs).map(|i| format!("__graph_{i}")).collect();

        // Pad dummy slots.
        for _ in actual_bs..padded_bs {
            full_seq_lens.push(1);
            full_block_ids.push(vec![0]); // dummy block 0
            full_tokens_before.push(0);
            full_is_prefill.push(false);
            full_req_ids.push("__pad__".to_string());
        }

        let query_start_loc: Vec<usize> = (0..=padded_bs).collect();
        let q_lens = vec![1; padded_bs];

        Self::new(
            padded_bs,
            padded_bs, // total_tokens = padded_bs (all decode, 1 token each)
            query_start_loc,
            q_lens,
            full_seq_lens,
            full_block_ids,
            full_tokens_before,
            full_is_prefill,
            full_req_ids,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> AttentionMetadata {
        // Batch: req0 is prefill with 5 tokens, req1 is decode with 1 token.
        AttentionMetadata::new(
            2,
            6,
            vec![0, 5, 6],
            vec![5, 1],
            vec![5, 10],
            vec![vec![0, 1], vec![2, 3]],
            vec![0, 9],
            vec![true, false],
            vec!["req-0".into(), "req-1".into()],
        )
    }

    #[test]
    fn test_request_slice() {
        let meta = sample_metadata();
        assert_eq!(meta.request_slice(0), (0, 5));
        assert_eq!(meta.request_slice(1), (5, 1));
    }

    #[test]
    fn test_is_all_decode() {
        let meta = sample_metadata();
        assert!(!meta.is_all_decode());

        let all_decode = AttentionMetadata::new(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 1],
            vec![10, 20],
            vec![vec![0], vec![1]],
            vec![9, 19],
            vec![false, false],
            vec!["a".into(), "b".into()],
        );
        assert!(all_decode.is_all_decode());
    }

    #[test]
    fn test_padded_decode() {
        let meta = AttentionMetadata::padded_decode(
            8,
            3,
            &[10, 20, 5],
            &[vec![0, 1], vec![2, 3, 4], vec![5]],
            &[9, 19, 4],
        );
        assert_eq!(meta.num_reqs, 8);
        assert_eq!(meta.total_tokens, 8);
        assert!(meta.is_all_decode());
        assert_eq!(meta.q_lens, vec![1; 8]);
        assert_eq!(meta.query_start_loc, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        // Real requests.
        assert_eq!(meta.seq_lens[0], 10);
        assert_eq!(meta.seq_lens[1], 20);
        assert_eq!(meta.seq_lens[2], 5);
        // Padded slots.
        for i in 3..8 {
            assert_eq!(meta.seq_lens[i], 1);
            assert_eq!(meta.block_ids[i], vec![0]);
            assert_eq!(meta.tokens_before[i], 0);
            assert_eq!(meta.req_ids[i], "__pad__");
        }
    }

    #[test]
    fn test_padded_decode_exact_size() {
        // actual_bs == padded_bs: no padding needed.
        let meta = AttentionMetadata::padded_decode(2, 2, &[10, 20], &[vec![0], vec![1]], &[9, 19]);
        assert_eq!(meta.num_reqs, 2);
        assert!(meta.is_all_decode());
    }

    #[test]
    fn test_query_start_loc_consistency() {
        let meta = sample_metadata();
        assert_eq!(meta.query_start_loc.len(), meta.num_reqs + 1);
        assert_eq!(*meta.query_start_loc.last().unwrap(), meta.total_tokens);
        for i in 0..meta.num_reqs {
            assert_eq!(
                meta.query_start_loc[i + 1] - meta.query_start_loc[i],
                meta.q_lens[i]
            );
        }
    }
}
