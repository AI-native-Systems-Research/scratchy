// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! High-level KV cache manager.
//!
//! Ported from `vllm/v1/core/kv_cache_manager.py`.  The `KVCacheManager`
//! wraps a [`BlockPool`] and provides the request-level API used by the
//! scheduler: looking up prefix-cache hits, allocating token slots, and
//! freeing blocks when a request finishes.
//!
//! This is a simplified initial port that handles the common single-group
//! (uniform attention) case.  Multi-group (hybrid) models, cross-attention,
//! and sliding-window coordinators will be added in a later phase.

use hashbrown::HashMap;

use scratchy_core_common::Request;
use scratchy_core_config::KVCacheConfig;

use crate::block_pool::BlockPool;
use crate::kv_cache_block::BlockHash;

/// Ceiling division: `cdiv(a, b) = ceil(a / b)`.
fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

// ---------------------------------------------------------------------------
// KVCacheManager
// ---------------------------------------------------------------------------

/// High-level KV cache manager that sits between the scheduler and the
/// [`BlockPool`].
///
/// It tracks per-request block allocations and provides an API for:
/// - Prefix-cache lookups ([`get_computed_blocks`])
/// - Slot allocation ([`allocate_slots`])
/// - Freeing blocks ([`free`])
///
/// Ported from `KVCacheManager` in `vllm/v1/core/kv_cache_manager.py`.
#[derive(Debug)]
pub struct KVCacheManager {
    /// The underlying block pool.
    block_pool: BlockPool,

    /// Whether prefix caching is enabled.
    enable_caching: bool,

    /// Maximum model context length in tokens.
    max_model_len: usize,

    /// Number of KV cache groups.
    num_kv_cache_groups: usize,

    /// Block size per group (tokens per block).
    block_sizes: Vec<usize>,

    /// Per-request block allocations.
    ///
    /// `req_to_blocks[request_id][group_idx]` is the list of arena block
    /// indices for that request in that cache group.
    req_to_blocks: HashMap<String, Vec<Vec<usize>>>,

    /// Per-request count of blocks that have already been cached
    /// (i.e. their block hash has been written into the prefix cache).
    req_to_num_cached_blocks: HashMap<String, usize>,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    ///
    /// # Arguments
    /// * `kv_cache_config` -- the KV cache configuration describing block
    ///   count and cache groups.
    /// * `max_model_len` -- the maximum context length the model supports.
    /// * `enable_caching` -- whether prefix caching is enabled.
    pub fn new(
        kv_cache_config: &KVCacheConfig,
        max_model_len: usize,
        enable_caching: bool,
    ) -> Self {
        let num_kv_cache_groups = kv_cache_config.kv_cache_groups.len();
        assert!(
            num_kv_cache_groups > 0,
            "kv_cache_config must have at least one cache group"
        );

        let block_sizes: Vec<usize> = kv_cache_config
            .kv_cache_groups
            .iter()
            .map(|g| g.kv_cache_spec.block_size)
            .collect();

        // All groups share the same hash block size (the minimum block size).
        let hash_block_size = *block_sizes.iter().min().unwrap();

        let block_pool =
            BlockPool::new(kv_cache_config.num_blocks, enable_caching, hash_block_size);

        Self {
            block_pool,
            enable_caching,
            max_model_len,
            num_kv_cache_groups,
            block_sizes,
            req_to_blocks: HashMap::new(),
            req_to_num_cached_blocks: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    pub fn usage(&self) -> f64 {
        self.block_pool.get_usage()
    }

    /// Number of free blocks in the pool.
    pub fn get_num_free_blocks(&self) -> usize {
        self.block_pool.get_num_free_blocks()
    }

    /// Get the block IDs for a request across all cache groups.
    ///
    /// Returns `None` if the request is not tracked.
    pub fn get_block_ids(&self, request_id: &str) -> Option<Vec<Vec<usize>>> {
        self.req_to_blocks.get(request_id).cloned()
    }

    /// Find prefix-cache hits for a request.
    ///
    /// Returns `(computed_blocks, num_computed_tokens)` where:
    /// - `computed_blocks[group_idx]` is a list of block indices that were
    ///   found in the prefix cache.
    /// - `num_computed_tokens` is the number of tokens covered by those
    ///   blocks.
    ///
    /// When caching is disabled, returns empty blocks and 0 tokens.
    pub fn get_computed_blocks(
        &mut self,
        request: &Request,
        request_block_hashes: &[BlockHash],
    ) -> (Vec<Vec<usize>>, usize) {
        if !self.enable_caching {
            return (self.empty_blocks(), 0);
        }

        let block_size = self.block_sizes[0]; // common case: single group
        // Maximum prefix we may cache-hit: prompt_length - 1 (we always need
        // to recompute at least the last token to get logits).
        let max_cache_hit_length = request.num_tokens().saturating_sub(1);
        let max_cache_hit_blocks = max_cache_hit_length / block_size;

        // Use a stack-allocated SmallVec for the common single-group case.
        let kv_cache_group_ids: smallvec::SmallVec<[u32; 4]> =
            (0..self.num_kv_cache_groups as u32).collect();

        let mut computed_blocks: Vec<Vec<usize>> =
            (0..self.num_kv_cache_groups).map(|_| Vec::new()).collect();
        let mut num_computed_tokens = 0usize;

        for (block_idx, hash) in request_block_hashes.iter().enumerate() {
            if block_idx >= max_cache_hit_blocks {
                break;
            }

            match self.block_pool.get_cached_block(hash, &kv_cache_group_ids) {
                Some(cached_indices) => {
                    for (group_idx, &blk_idx) in cached_indices.iter().enumerate() {
                        computed_blocks[group_idx].push(blk_idx);
                    }
                    num_computed_tokens = (block_idx + 1) * block_size;
                }
                None => break, // prefix cache is contiguous; stop on first miss.
            }
        }

        (computed_blocks, num_computed_tokens)
    }

    /// Allocate token slots for a request.
    ///
    /// # Arguments
    /// * `request` -- the request to allocate for.
    /// * `request_block_hashes` -- the block hashes for prefix caching.
    /// * `num_new_tokens` -- number of new tokens to be computed.
    /// * `num_new_computed_tokens` -- number of new tokens from prefix-cache
    ///   hits.
    /// * `new_computed_blocks` -- cached block indices from
    ///   `get_computed_blocks`.
    ///
    /// Returns `Some(new_block_ids)` on success, or `None` if there are not
    /// enough free blocks.
    pub fn allocate_slots(
        &mut self,
        request: &Request,
        request_block_hashes: &[BlockHash],
        num_new_tokens: usize,
        num_new_computed_tokens: usize,
        new_computed_blocks: &[Vec<usize>],
    ) -> Option<Vec<Vec<usize>>> {
        if num_new_tokens == 0 {
            return Some(self.empty_blocks());
        }

        let block_size = self.block_sizes[0];

        let num_computed_tokens = request.num_computed_tokens as usize + num_new_computed_tokens;
        // ⭐ SLOTS, NOT TOKENS, WHEN THE WORKER SAYS THEY DIFFER. A backend that appends a whole batch at
        // ONE KV slot per step leaves a masked hole in every request shorter than
        // its batch-mates, so that request's keys occupy MORE slots than it has tokens and blocks sized by
        // the token count stop short of the slot the next forward writes. The worker reports the span it
        // actually occupies (`ModelRunnerOutput::kv_extent`); `None` is every backend whose keys sit at
        // their token positions, and there this is exactly the old expression.
        //
        // ⛔ PLUS THIS STEP'S TOKENS. The span is where the request's keys REACH; the forward about to run
        // writes `num_new_tokens` more AT OR PAST it, so blocks sized by the span alone stop one slot
        // short — every time the span lands exactly on a block boundary, which for a decode loop is once
        // every `block_size` steps and always mid-generation.
        let num_slots_need_block = request
            .kv_extent
            .map_or(0, |e| e.span().get() as usize + num_new_tokens)
            .max(num_computed_tokens + num_new_tokens);
        let num_tokens_need_slot = std::cmp::min(num_slots_need_block, self.max_model_len);

        // Initialize per-request tracking if this is the first allocation.
        let req_blocks = self
            .req_to_blocks
            .entry(request.request_id.clone())
            .or_insert_with(|| (0..self.num_kv_cache_groups).map(|_| Vec::new()).collect());

        // Append the new prefix-cached blocks and update the cached-blocks
        // counter so we never try to re-hash them.
        let num_new_cached = new_computed_blocks.first().map(|g| g.len()).unwrap_or(0);
        for (group_idx, cached) in new_computed_blocks.iter().enumerate() {
            req_blocks[group_idx].extend_from_slice(cached);
        }

        if self.enable_caching && num_new_cached > 0 {
            // Prefix-cached blocks already have their hashes set in the block
            // pool. Record them so cache_full_blocks() skips over them.
            let entry = self
                .req_to_num_cached_blocks
                .entry(request.request_id.clone())
                .or_insert(0);
            *entry += num_new_cached;
        }

        // Touch the prefix-cached blocks to bump their ref counts.
        if self.enable_caching {
            for cached in new_computed_blocks {
                self.block_pool.touch(cached);
            }
        }

        // How many blocks does the request need in total?
        let num_required_blocks = cdiv(num_tokens_need_slot, block_size);
        let num_current_blocks = self.req_to_blocks[&request.request_id][0].len();
        let num_blocks_to_allocate = num_required_blocks.saturating_sub(num_current_blocks);

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks() {
            return None;
        }

        // Allocate new blocks.
        let new_block_indices = if num_blocks_to_allocate > 0 {
            self.block_pool.get_new_blocks(num_blocks_to_allocate)
        } else {
            Vec::new()
        };

        // Append new blocks to every cache group for this request.
        // In the common single-group case, all groups share the same block IDs.
        let req_blocks = self
            .req_to_blocks
            .get_mut(&request.request_id)
            .expect("request must be tracked");
        for group in req_blocks.iter_mut() {
            group.extend_from_slice(&new_block_indices);
        }

        // Cache full blocks if prefix caching is enabled.
        if self.enable_caching {
            let num_cached_blocks = *self
                .req_to_num_cached_blocks
                .entry(request.request_id.clone())
                .or_insert(0);

            // ⛔ AND ONLY THE PREFIX WHOSE KEYS ARE AT THEIR TOKEN POSITIONS MAY BE HASHED. A cached block
            // is a promise to a LATER request that block `b` holds the keys of tokens
            // `[b*block_size, (b+1)*block_size)`. On a backend where a request's keys spread past its
            // token count (see `num_slots_need_block`) that promise holds for the leading contiguous run
            // and for nothing after it: past the first hole, token `t` is at some slot `> t`, and a
            // reusing request would attend the keys of the wrong tokens — fluent, wrong, no fault. The
            // worker reports how far the promise reaches; `None` means the whole computed prefix, which is
            // every backend whose KV is token-addressed.
            let cacheable = request
                .kv_extent
                .map_or(usize::MAX, |e| e.cacheable_tokens().get() as usize);
            let num_tokens_to_cache = std::cmp::min(
                std::cmp::min(num_computed_tokens + num_new_tokens, request.num_tokens()),
                cacheable,
            );
            let num_full_blocks = num_tokens_to_cache / block_size;

            if num_full_blocks > num_cached_blocks {
                let all_blocks = &self.req_to_blocks[&request.request_id];
                for (group_idx, group_blocks) in
                    all_blocks.iter().enumerate().take(self.num_kv_cache_groups)
                {
                    self.block_pool.cache_full_blocks(
                        request_block_hashes,
                        group_blocks,
                        num_cached_blocks,
                        num_full_blocks,
                        block_size,
                        group_idx as u32,
                    );
                }
                self.req_to_num_cached_blocks
                    .insert(request.request_id.clone(), num_full_blocks);
            }
        }

        // Build the return value: only the *new* blocks.
        // For the common single-group case, move instead of clone.
        if self.num_kv_cache_groups == 1 {
            return Some(vec![new_block_indices]);
        }
        let mut new_blocks_per_group: Vec<Vec<usize>> = (0..self.num_kv_cache_groups - 1)
            .map(|_| new_block_indices.clone())
            .collect();
        new_blocks_per_group.push(new_block_indices); // move the last one
        Some(new_blocks_per_group)
    }

    /// Free all blocks allocated for a request.
    ///
    /// Blocks are freed in reverse order so that tail blocks (which are more
    /// recently cached) are evicted first when the free list is consumed,
    /// maintaining LRU order.
    pub fn free(&mut self, request_id: &str) {
        if let Some(groups) = self.req_to_blocks.remove(request_id) {
            for group in &groups {
                // Free in reverse order for LRU eviction.
                // Use SmallVec to avoid heap allocation for typical block counts.
                let reversed: smallvec::SmallVec<[usize; 16]> =
                    group.iter().copied().rev().collect();
                self.block_pool.free_blocks(&reversed);
            }
        }
        self.req_to_num_cached_blocks.remove(request_id);
    }

    /// Reset the prefix cache.
    ///
    /// Returns `true` on success, `false` if blocks are still in use.
    pub fn reset_prefix_cache(&mut self) -> bool {
        if !self.block_pool.reset_prefix_cache() {
            return false;
        }
        self.req_to_num_cached_blocks.clear();
        true
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Create an empty per-group block list.
    fn empty_blocks(&self) -> Vec<Vec<usize>> {
        (0..self.num_kv_cache_groups).map(|_| Vec::new()).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_core_common::SamplingParams;
    use scratchy_core_config::{
        KVCacheConfig, KVCacheGroupSpec, KVCacheSpec, KVCacheSpecType, KVCacheTensor,
    };

    /// Helper to build a KVCacheConfig with one full-attention group.
    fn make_config(num_blocks: usize, block_size: usize) -> KVCacheConfig {
        KVCacheConfig {
            num_blocks,
            kv_cache_tensors: vec![KVCacheTensor {
                size: 1024,
                shared_by: vec!["layer0".into()],
            }],
            kv_cache_groups: vec![KVCacheGroupSpec {
                layer_names: vec!["layer0".into()],
                kv_cache_spec: KVCacheSpec {
                    block_size,
                    spec_type: KVCacheSpecType::FullAttention {
                        num_kv_heads: 8,
                        head_size: 128,
                        head_size_v: None,
                        sliding_window: None,
                        attention_chunk_size: None,
                    },
                },
            }],
        }
    }

    /// Helper to build a Request.
    fn make_request(id: &str, prompt_tokens: &[u32]) -> Request {
        Request::new(
            id.into(),
            prompt_tokens.to_vec(),
            SamplingParams::default(),
            0.0,
            0,
            0,
            None,
        )
    }

    /// Helper to create fake block hashes for a given number of blocks.
    fn make_block_hashes(num_blocks: usize) -> Vec<BlockHash> {
        (0..num_blocks)
            .map(|i| {
                let mut h = vec![0u8; 32];
                h[0] = i as u8;
                h[1] = (i >> 8) as u8;
                h
            })
            .collect()
    }

    #[test]
    fn test_new_manager() {
        let cfg = make_config(100, 16);
        let mgr = KVCacheManager::new(&cfg, 2048, true);
        assert_eq!(mgr.get_num_free_blocks(), 99); // 100 - 1 null
        assert!((mgr.usage() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_allocate_slots_basic() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, false);

        // 32-token prompt => 2 blocks.
        let prompt: Vec<u32> = (0..32).collect();
        let req = make_request("r1", &prompt);
        let hashes = make_block_hashes(2);

        let result = mgr.allocate_slots(&req, &hashes, 32, 0, &[vec![]]);
        assert!(result.is_some());
        let new_blocks = result.unwrap();
        assert_eq!(new_blocks.len(), 1); // one group
        assert_eq!(new_blocks[0].len(), 2); // 2 blocks allocated

        // Check the request is tracked.
        let block_ids = mgr.get_block_ids("r1");
        assert!(block_ids.is_some());
        assert_eq!(block_ids.unwrap()[0].len(), 2);
    }

    #[test]
    fn test_allocate_slots_returns_none_on_oom() {
        let cfg = make_config(5, 16); // 4 free blocks (5 - 1 null).
        let mut mgr = KVCacheManager::new(&cfg, 2048, false);

        // 80 tokens => 5 blocks needed, but only 4 free.
        let prompt: Vec<u32> = (0..80).collect();
        let req = make_request("r1", &prompt);
        let hashes = make_block_hashes(5);

        let result = mgr.allocate_slots(&req, &hashes, 80, 0, &[vec![]]);
        assert!(result.is_none());
    }

    #[test]
    fn test_free_request() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, false);

        let prompt: Vec<u32> = (0..32).collect();
        let req = make_request("r1", &prompt);
        let hashes = make_block_hashes(2);

        mgr.allocate_slots(&req, &hashes, 32, 0, &[vec![]]);
        assert_eq!(mgr.get_num_free_blocks(), 97); // 99 - 2

        mgr.free("r1");
        assert_eq!(mgr.get_num_free_blocks(), 99);
        assert!(mgr.get_block_ids("r1").is_none());
    }

    #[test]
    fn test_get_computed_blocks_caching_disabled() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, false);

        let req = make_request("r1", &(0..64).collect::<Vec<_>>());
        let hashes = make_block_hashes(4);

        let (blocks, num_tokens) = mgr.get_computed_blocks(&req, &hashes);
        assert_eq!(num_tokens, 0);
        assert!(blocks.iter().all(|g| g.is_empty()));
    }

    #[test]
    fn test_prefix_cache_hit() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, true);

        // First request: 64 tokens => 4 blocks.
        let prompt1: Vec<u32> = (0..64).collect();
        let req1 = make_request("r1", &prompt1);
        let hashes = make_block_hashes(4);

        // Allocate -- this also caches the full blocks.
        mgr.allocate_slots(&req1, &hashes, 64, 0, &[vec![]]);

        // Free request 1. Blocks go back to the free list but remain cached.
        mgr.free("r1");

        // Second request with the same prefix.
        let prompt2: Vec<u32> = (0..80).collect(); // 64 common + 16 new
        let req2 = make_request("r2", &prompt2);
        let hashes2 = make_block_hashes(5);

        let (computed_blocks, num_computed_tokens) = mgr.get_computed_blocks(&req2, &hashes2);

        // max_cache_hit_length = 80 - 1 = 79 => max 4 blocks (79/16 = 4)
        // We should hit 4 cached blocks.
        assert_eq!(num_computed_tokens, 64);
        assert_eq!(computed_blocks[0].len(), 4);
    }

    #[test]
    fn test_prefix_cache_partial_hit() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, true);

        // First request: 32 tokens => 2 blocks cached.
        let prompt1: Vec<u32> = (0..32).collect();
        let req1 = make_request("r1", &prompt1);
        let hashes = make_block_hashes(4); // we have 4 hashes, but only 2 blocks cached

        mgr.allocate_slots(&req1, &hashes, 32, 0, &[vec![]]);
        mgr.free("r1");

        // Second request with 4 block hashes, but only the first 2 are cached.
        let prompt2: Vec<u32> = (0..64).collect();
        let req2 = make_request("r2", &prompt2);
        let hashes2 = make_block_hashes(4);

        let (computed_blocks, num_computed_tokens) = mgr.get_computed_blocks(&req2, &hashes2);

        assert_eq!(num_computed_tokens, 32); // only 2 blocks hit
        assert_eq!(computed_blocks[0].len(), 2);
    }

    #[test]
    fn test_allocate_with_prefix_cache_hit() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, true);

        // First request: cache 2 blocks.
        let prompt1: Vec<u32> = (0..32).collect();
        let req1 = make_request("r1", &prompt1);
        let hashes = make_block_hashes(4);

        mgr.allocate_slots(&req1, &hashes, 32, 0, &[vec![]]);
        mgr.free("r1");

        // Second request: look up prefix.
        let prompt2: Vec<u32> = (0..48).collect();
        let req2 = make_request("r2", &prompt2);
        let hashes2 = make_block_hashes(3);

        let (computed_blocks, num_computed_tokens) = mgr.get_computed_blocks(&req2, &hashes2);
        assert_eq!(num_computed_tokens, 32);

        // Now allocate, passing the prefix blocks.
        let mut req2_for_alloc = req2.clone();
        req2_for_alloc.num_computed_tokens = 0; // not yet computed
        let result = mgr.allocate_slots(
            &req2_for_alloc,
            &hashes2,
            48 - num_computed_tokens, // new tokens to compute
            num_computed_tokens,
            &computed_blocks,
        );
        assert!(result.is_some());
        let new_blocks = result.unwrap();
        // 3 blocks needed total, 2 from cache => 1 new block.
        assert_eq!(new_blocks[0].len(), 1);

        // The request should now have 3 blocks total.
        let all_blocks = mgr.get_block_ids("r2").unwrap();
        assert_eq!(all_blocks[0].len(), 3);
    }

    #[test]
    fn test_reset_prefix_cache() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, true);
        assert!(mgr.reset_prefix_cache());
    }

    #[test]
    fn test_reset_prefix_cache_fails_with_active_requests() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, true);

        let req = make_request("r1", &(0..16).collect::<Vec<_>>());
        let hashes = make_block_hashes(1);
        mgr.allocate_slots(&req, &hashes, 16, 0, &[vec![]]);

        assert!(!mgr.reset_prefix_cache());
    }

    #[test]
    fn test_multiple_requests() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, false);

        let req1 = make_request("r1", &(0..32).collect::<Vec<_>>());
        let req2 = make_request("r2", &(0..48).collect::<Vec<_>>());
        let hashes1 = make_block_hashes(2);
        let hashes2 = make_block_hashes(3);

        mgr.allocate_slots(&req1, &hashes1, 32, 0, &[vec![]]);
        mgr.allocate_slots(&req2, &hashes2, 48, 0, &[vec![]]);

        assert_eq!(mgr.get_num_free_blocks(), 94); // 99 - 2 - 3

        mgr.free("r1");
        assert_eq!(mgr.get_num_free_blocks(), 96); // 94 + 2

        mgr.free("r2");
        assert_eq!(mgr.get_num_free_blocks(), 99);
    }

    #[test]
    fn test_incremental_allocation() {
        let cfg = make_config(100, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, false);

        // First allocation: 16 tokens => 1 block.
        let prompt: Vec<u32> = (0..16).collect();
        let mut req = make_request("r1", &prompt);
        let hashes = make_block_hashes(1);

        let result = mgr.allocate_slots(&req, &hashes, 16, 0, &[vec![]]);
        assert!(result.is_some());
        assert_eq!(mgr.get_block_ids("r1").unwrap()[0].len(), 1);

        // Second allocation: append 16 more tokens => need 2 blocks total.
        req.append_output_token_ids(&(16..32).collect::<Vec<_>>());
        req.num_computed_tokens = 16;
        let hashes2 = make_block_hashes(2);

        let result = mgr.allocate_slots(&req, &hashes2, 16, 0, &[vec![]]);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0].len(), 1); // 1 new block
        assert_eq!(mgr.get_block_ids("r1").unwrap()[0].len(), 2); // 2 total
    }

    #[test]
    fn test_free_nonexistent_request_is_noop() {
        let cfg = make_config(10, 16);
        let mut mgr = KVCacheManager::new(&cfg, 2048, false);
        mgr.free("nonexistent"); // should not panic
    }
}
