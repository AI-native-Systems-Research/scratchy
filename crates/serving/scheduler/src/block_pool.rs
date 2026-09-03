// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Block pool for KV cache block management.
//!
//! Ported from `vllm/v1/core/block_pool.py`.  The `BlockPool` owns the
//! flat arena of [`KVCacheBlock`]s, the [`FreeKVCacheBlockQueue`], and the
//! prefix-cache hash map.

use hashbrown::HashMap;
use smallvec::SmallVec;
use tracing::warn;

use crate::free_block_queue::FreeKVCacheBlockQueue;
use crate::kv_cache_block::{
    BlockHash, BlockHashWithGroupId, KVCacheBlock, make_block_hash_with_group_id,
};

// ---------------------------------------------------------------------------
// BlockHashToBlockMap
// ---------------------------------------------------------------------------

/// Maps `BlockHashWithGroupId` -> set of arena block indices that share that
/// hash.
///
/// Most hashes map to exactly one block, so we use `SmallVec<[usize; 1]>` to
/// avoid a heap allocation in the common case.
///
/// Ported from `BlockHashToBlockMap` in `block_pool.py`.
#[derive(Debug, Default)]
pub struct BlockHashToBlockMap {
    cache: HashMap<BlockHashWithGroupId, SmallVec<[usize; 1]>>,
}

impl BlockHashToBlockMap {
    /// Get any one block index for the given hash, or `None` on cache miss.
    pub fn get_one_block(&self, key: &BlockHashWithGroupId) -> Option<usize> {
        self.cache.get(key).and_then(|v| v.first().copied())
    }

    /// Insert a block index under the given hash key.
    pub fn insert(&mut self, key: BlockHashWithGroupId, block_idx: usize) {
        self.cache.entry(key).or_default().push(block_idx);
    }

    /// Remove `block_idx` from the entry for `key`.
    ///
    /// Returns `Some(block_idx)` if found and removed, `None` otherwise.
    /// If the entry becomes empty after removal it is deleted from the map.
    pub fn pop(&mut self, key: &BlockHashWithGroupId, block_idx: usize) -> Option<usize> {
        let entry = self.cache.get_mut(key)?;
        if let Some(pos) = entry.iter().position(|&idx| idx == block_idx) {
            let removed = entry.swap_remove(pos);
            if entry.is_empty() {
                self.cache.remove(key);
            }
            Some(removed)
        } else {
            None
        }
    }

    /// Number of distinct hash keys in the map.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

// ---------------------------------------------------------------------------
// BlockPool
// ---------------------------------------------------------------------------

/// Manages an arena of [`KVCacheBlock`]s, a free-block queue, and a
/// prefix-cache hash map.
///
/// Ported from `BlockPool` in `vllm/v1/core/block_pool.py`.
///
/// # Arena layout
///
/// ```text
/// [0 .. num_gpu_blocks-1]  -- real blocks
/// [num_gpu_blocks]         -- head sentinel for free-block queue
/// [num_gpu_blocks+1]       -- tail sentinel for free-block queue
/// ```
///
/// Block 0 is reserved as the *null block* (placeholder that is never freed
/// or cached).
#[derive(Debug)]
pub struct BlockPool {
    /// Total number of real (non-sentinel) blocks.
    num_gpu_blocks: usize,

    /// Whether prefix caching is enabled.
    enable_caching: bool,

    /// The block size used for hashing.
    pub hash_block_size: usize,

    /// Flat arena: real blocks + 2 sentinel blocks.
    blocks: Vec<KVCacheBlock>,

    /// Doubly-linked free-block queue (indices into `blocks`).
    free_block_queue: FreeKVCacheBlockQueue,

    /// Prefix-cache hash map.
    cached_block_hash_to_block: BlockHashToBlockMap,

    /// Index of the null block (always 0).
    null_block_idx: usize,
}

impl BlockPool {
    /// Create a new block pool.
    ///
    /// # Arguments
    /// * `num_gpu_blocks` -- number of real blocks to manage (must be > 0).
    /// * `enable_caching` -- whether prefix caching is enabled.
    /// * `hash_block_size` -- the block size at which block hashes are computed.
    ///
    /// # Panics
    /// Panics if `num_gpu_blocks == 0`.
    pub fn new(num_gpu_blocks: usize, enable_caching: bool, hash_block_size: usize) -> Self {
        assert!(num_gpu_blocks > 0, "num_gpu_blocks must be > 0");

        let head_sentinel_idx = num_gpu_blocks;
        let tail_sentinel_idx = num_gpu_blocks + 1;

        // Build arena: real blocks + 2 sentinels.
        let mut blocks: Vec<KVCacheBlock> = (0..num_gpu_blocks).map(KVCacheBlock::new).collect();
        blocks.push(KVCacheBlock::new(usize::MAX - 1)); // head sentinel
        blocks.push(KVCacheBlock::new(usize::MAX)); // tail sentinel

        // Initialize the free-block queue with all real blocks.
        let mut free_block_queue = FreeKVCacheBlockQueue::new(
            &mut blocks,
            num_gpu_blocks,
            head_sentinel_idx,
            tail_sentinel_idx,
        );

        // Pop block 0 as the null block.
        let null_block_idx = free_block_queue.popleft(&mut blocks);
        debug_assert_eq!(null_block_idx, 0);
        blocks[null_block_idx].is_null = true;

        Self {
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            blocks,
            free_block_queue,
            cached_block_hash_to_block: BlockHashToBlockMap::default(),
            null_block_idx,
        }
    }

    /// Get the arena index of the null block.
    pub fn null_block_idx(&self) -> usize {
        self.null_block_idx
    }

    /// Read-only access to a block by arena index.
    pub fn block(&self, idx: usize) -> &KVCacheBlock {
        &self.blocks[idx]
    }

    /// Mutable access to a block by arena index.
    pub fn block_mut(&mut self, idx: usize) -> &mut KVCacheBlock {
        &mut self.blocks[idx]
    }

    // -----------------------------------------------------------------------
    // Prefix-cache lookup
    // -----------------------------------------------------------------------

    /// Find cached blocks for the given `block_hash` across all groups in
    /// `kv_cache_group_ids`.
    ///
    /// Returns `Some(vec_of_block_indices)` if every group has a cached block,
    /// or `None` if any group is a cache miss.
    pub fn get_cached_block(
        &self,
        block_hash: &BlockHash,
        kv_cache_group_ids: &[u32],
    ) -> Option<Vec<usize>> {
        let mut cached = Vec::with_capacity(kv_cache_group_ids.len());
        for &group_id in kv_cache_group_ids {
            let key = make_block_hash_with_group_id(block_hash, group_id);
            cached.push(self.cached_block_hash_to_block.get_one_block(&key)?);
        }
        Some(cached)
    }

    // -----------------------------------------------------------------------
    // Caching full blocks
    // -----------------------------------------------------------------------

    /// Cache a range of full blocks for prefix caching.
    ///
    /// Updates the block-hash metadata for blocks `[num_cached_blocks ..
    /// num_full_blocks)` and inserts them into the prefix-cache map.
    ///
    /// # Arguments
    /// * `request_block_hashes` -- the ordered block hashes for the request
    ///   (one per block at `hash_block_size` granularity).
    /// * `block_indices` -- the arena indices of all blocks for this request
    ///   (in the relevant cache group).
    /// * `num_cached_blocks` -- how many blocks are already cached.
    /// * `num_full_blocks` -- total number of full blocks after this call.
    /// * `block_size` -- the actual block size for this cache group (may be a
    ///   multiple of `hash_block_size`).
    /// * `kv_cache_group_id` -- the cache group id.
    pub fn cache_full_blocks(
        &mut self,
        request_block_hashes: &[BlockHash],
        block_indices: &[usize],
        num_cached_blocks: usize,
        num_full_blocks: usize,
        block_size: usize,
        kv_cache_group_id: u32,
    ) {
        if num_cached_blocks >= num_full_blocks {
            return;
        }

        // When block_size == hash_block_size we can use the hashes directly;
        // otherwise we would need to recombine them.  For this initial port
        // we support the common case and assert equality.
        assert_eq!(
            block_size, self.hash_block_size,
            "block_size ({}) must equal hash_block_size ({}) in this port",
            block_size, self.hash_block_size,
        );

        assert!(
            request_block_hashes.len() >= num_full_blocks,
            "Not enough block hashes ({}) for num_full_blocks ({})",
            request_block_hashes.len(),
            num_full_blocks,
        );

        for i in num_cached_blocks..num_full_blocks {
            let blk_idx = block_indices[i];

            // Skip null blocks.
            if self.blocks[blk_idx].is_null {
                continue;
            }

            // The block should not already be cached.
            assert!(
                self.blocks[blk_idx].block_hash().is_none(),
                "Block {} already has a hash",
                blk_idx,
            );

            let block_hash = &request_block_hashes[i];
            let key = make_block_hash_with_group_id(block_hash, kv_cache_group_id);
            self.blocks[blk_idx].set_block_hash(key.clone());
            self.cached_block_hash_to_block.insert(key, blk_idx);
        }
    }

    // -----------------------------------------------------------------------
    // Allocation
    // -----------------------------------------------------------------------

    /// Allocate `num_blocks` new blocks from the free pool.
    ///
    /// If caching is enabled, evicts cached metadata from any blocks that are
    /// being recycled.
    ///
    /// Returns a `Vec` of arena indices.
    ///
    /// # Panics
    /// Panics if there are not enough free blocks.
    pub fn get_new_blocks(&mut self, num_blocks: usize) -> Vec<usize> {
        assert!(
            num_blocks <= self.get_num_free_blocks(),
            "Cannot allocate {} blocks; only {} free",
            num_blocks,
            self.get_num_free_blocks(),
        );

        let indices = self
            .free_block_queue
            .popleft_n(&mut self.blocks, num_blocks);

        for &idx in &indices {
            if self.enable_caching {
                self.maybe_evict_cached_block(idx);
            }
            debug_assert_eq!(self.blocks[idx].ref_cnt, 0);
            self.blocks[idx].ref_cnt = 1;
        }

        indices
    }

    // -----------------------------------------------------------------------
    // Touch / Free
    // -----------------------------------------------------------------------

    /// Touch (increment ref_cnt of) blocks that were found in the prefix
    /// cache.  If a block has `ref_cnt == 0` it is currently in the free list
    /// and must be removed from it.
    pub fn touch(&mut self, block_indices: &[usize]) {
        for &idx in block_indices {
            if self.blocks[idx].ref_cnt == 0 && !self.blocks[idx].is_null {
                self.free_block_queue.remove(&mut self.blocks, idx);
            }
            self.blocks[idx].ref_cnt += 1;
        }
    }

    /// Free a list of blocks.  Decrements each block's ref_cnt and, if it
    /// reaches zero, appends the block back to the free list.
    ///
    /// The `block_indices` should be ordered by eviction priority (the first
    /// block will be evicted first when the free list is consumed).
    pub fn free_blocks(&mut self, block_indices: &[usize]) {
        // First pass: decrement ref counts.
        for &idx in block_indices {
            self.blocks[idx].ref_cnt -= 1;
        }

        // Second pass: collect blocks with ref_cnt == 0 that should go back
        // to the free list. Use SmallVec to avoid heap allocation for typical cases.
        let to_free: SmallVec<[usize; 16]> = block_indices
            .iter()
            .copied()
            .filter(|&idx| self.blocks[idx].ref_cnt == 0 && !self.blocks[idx].is_null)
            .collect();

        self.free_block_queue.append_n(&mut self.blocks, &to_free);
    }

    // -----------------------------------------------------------------------
    // Eviction helpers
    // -----------------------------------------------------------------------

    /// If a block has cached hash metadata, evict it from the prefix cache.
    ///
    /// Returns `true` if eviction occurred.
    fn maybe_evict_cached_block(&mut self, block_idx: usize) -> bool {
        let block_hash = match self.blocks[block_idx].block_hash().cloned() {
            Some(h) => h,
            None => return false,
        };

        if self
            .cached_block_hash_to_block
            .pop(&block_hash, block_idx)
            .is_none()
        {
            return false;
        }

        self.blocks[block_idx].reset_hash();
        true
    }

    // -----------------------------------------------------------------------
    // Reset / stats
    // -----------------------------------------------------------------------

    /// Reset the prefix cache.  Only succeeds if all blocks (except the null
    /// block) are currently free.
    ///
    /// Returns `true` on success, `false` if blocks are still in use.
    pub fn reset_prefix_cache(&mut self) -> bool {
        let num_used = self.num_gpu_blocks - self.get_num_free_blocks();
        if num_used != 1 {
            // Only the null block should be "used".
            warn!(
                "Failed to reset prefix cache: {} blocks still in use",
                num_used - 1
            );
            return false;
        }

        self.cached_block_hash_to_block.clear();

        for block in self.blocks.iter_mut() {
            block.reset_hash();
        }

        true
    }

    /// Number of free blocks in the pool.
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_block_queue.num_free_blocks()
    }

    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    pub fn get_usage(&self) -> f64 {
        // Subtract 1 for the null block.
        let total = self.num_gpu_blocks - 1;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.get_num_free_blocks() as f64 / total as f64)
    }

    /// Total number of real blocks managed by the pool.
    pub fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_block_pool() {
        let pool = BlockPool::new(10, false, 16);
        // 10 blocks total, block 0 is null, so 9 free.
        assert_eq!(pool.get_num_free_blocks(), 9);
        assert_eq!(pool.null_block_idx(), 0);
        assert!(pool.block(0).is_null);
    }

    #[test]
    fn test_get_new_blocks_no_caching() {
        let mut pool = BlockPool::new(10, false, 16);
        let allocated = pool.get_new_blocks(3);
        assert_eq!(allocated.len(), 3);
        assert_eq!(pool.get_num_free_blocks(), 6);
        // Each allocated block should have ref_cnt == 1.
        for &idx in &allocated {
            assert_eq!(pool.block(idx).ref_cnt, 1);
        }
    }

    #[test]
    fn test_free_blocks() {
        let mut pool = BlockPool::new(10, false, 16);
        let allocated = pool.get_new_blocks(3);
        assert_eq!(pool.get_num_free_blocks(), 6);

        pool.free_blocks(&allocated);
        assert_eq!(pool.get_num_free_blocks(), 9);
        for &idx in &allocated {
            assert_eq!(pool.block(idx).ref_cnt, 0);
        }
    }

    #[test]
    fn test_touch_blocks() {
        let mut pool = BlockPool::new(10, true, 16);
        // Allocate 2 blocks.
        let allocated = pool.get_new_blocks(2);
        assert_eq!(pool.get_num_free_blocks(), 7);

        // Free them so they go back to the free list.
        pool.free_blocks(&allocated);
        assert_eq!(pool.get_num_free_blocks(), 9);

        // Touch them (simulating a cache hit).
        pool.touch(&allocated);
        // They should be removed from the free list.
        assert_eq!(pool.get_num_free_blocks(), 7);
        // ref_cnt should be 1 again.
        for &idx in &allocated {
            assert_eq!(pool.block(idx).ref_cnt, 1);
        }
    }

    #[test]
    fn test_touch_already_in_use() {
        let mut pool = BlockPool::new(10, true, 16);
        let allocated = pool.get_new_blocks(2);
        // Touch while already in use (ref_cnt > 0).
        pool.touch(&allocated);
        // ref_cnt should now be 2.
        for &idx in &allocated {
            assert_eq!(pool.block(idx).ref_cnt, 2);
        }
        // Free list count unchanged.
        assert_eq!(pool.get_num_free_blocks(), 7);
    }

    #[test]
    fn test_get_usage() {
        let mut pool = BlockPool::new(10, false, 16);
        // 9 usable blocks (10 - 1 null), 9 free => usage = 0.0.
        assert!((pool.get_usage() - 0.0).abs() < 1e-9);

        let _ = pool.get_new_blocks(9);
        // 0 free => usage = 1.0.
        assert!((pool.get_usage() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_usage_partial() {
        let mut pool = BlockPool::new(10, false, 16);
        let _ = pool.get_new_blocks(3);
        // 9 usable, 6 free => usage = 3/9.
        let expected = 3.0 / 9.0;
        assert!((pool.get_usage() - expected).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "Cannot allocate")]
    fn test_allocate_too_many_panics() {
        let mut pool = BlockPool::new(5, false, 16);
        // 4 free (5 - 1 null). Requesting 5 should panic.
        pool.get_new_blocks(5);
    }

    #[test]
    fn test_cache_and_lookup() {
        let mut pool = BlockPool::new(10, true, 16);
        let allocated = pool.get_new_blocks(3);

        // Create some fake block hashes.
        let hashes: Vec<BlockHash> = (0..3).map(|i| vec![i as u8; 32]).collect();

        // Cache blocks 0..3 with group id 0.
        pool.cache_full_blocks(&hashes, &allocated, 0, 3, 16, 0);

        // All three blocks should now be findable.
        for (i, hash) in hashes.iter().enumerate() {
            let found = pool.get_cached_block(hash, &[0]);
            assert!(found.is_some(), "block {} not found", i);
            assert_eq!(found.unwrap(), vec![allocated[i]]);
        }
    }

    #[test]
    fn test_cache_miss() {
        let pool = BlockPool::new(10, true, 16);
        let missing_hash: BlockHash = vec![0xFF; 32];
        assert!(pool.get_cached_block(&missing_hash, &[0]).is_none());
    }

    #[test]
    fn test_cache_multi_group() {
        let mut pool = BlockPool::new(20, true, 16);
        let blocks_g0 = pool.get_new_blocks(2);
        let blocks_g1 = pool.get_new_blocks(2);

        let hashes: Vec<BlockHash> = (0..2).map(|i| vec![i as u8; 32]).collect();

        // Cache for group 0.
        pool.cache_full_blocks(&hashes, &blocks_g0, 0, 2, 16, 0);
        // Cache for group 1.
        pool.cache_full_blocks(&hashes, &blocks_g1, 0, 2, 16, 1);

        // Looking up both groups should succeed.
        let found = pool.get_cached_block(&hashes[0], &[0, 1]);
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0], blocks_g0[0]);
        assert_eq!(found[1], blocks_g1[0]);
    }

    #[test]
    fn test_cache_partial_group_miss() {
        let mut pool = BlockPool::new(20, true, 16);
        let blocks_g0 = pool.get_new_blocks(1);

        let hash: BlockHash = vec![0xAA; 32];
        pool.cache_full_blocks(std::slice::from_ref(&hash), &blocks_g0, 0, 1, 16, 0);

        // group 0 hit, group 1 miss => None.
        let found = pool.get_cached_block(&hash, &[0, 1]);
        assert!(found.is_none());
    }

    #[test]
    fn test_eviction_on_allocation() {
        let mut pool = BlockPool::new(5, true, 16);
        // 4 free blocks.

        // Allocate all 4.
        let first_batch = pool.get_new_blocks(4);
        assert_eq!(pool.get_num_free_blocks(), 0);

        // Cache all 4 blocks.
        let hashes: Vec<BlockHash> = (0..4).map(|i| vec![i as u8; 16]).collect();
        pool.cache_full_blocks(&hashes, &first_batch, 0, 4, 16, 0);

        // Free all 4 (they go back to the free list as eviction candidates).
        pool.free_blocks(&first_batch);
        assert_eq!(pool.get_num_free_blocks(), 4);

        // Now allocate 2 new blocks -- they should recycle from the free list
        // and evict the cached hashes.
        let second_batch = pool.get_new_blocks(2);
        assert_eq!(pool.get_num_free_blocks(), 2);

        // The recycled blocks should have their hashes evicted.
        for &idx in &second_batch {
            assert!(pool.block(idx).block_hash().is_none());
        }
    }

    #[test]
    fn test_reset_prefix_cache_success() {
        let mut pool = BlockPool::new(5, true, 16);
        assert!(pool.reset_prefix_cache());
    }

    #[test]
    fn test_reset_prefix_cache_fail_blocks_in_use() {
        let mut pool = BlockPool::new(5, true, 16);
        let _ = pool.get_new_blocks(1);
        // Now 1 block in use + null block => 2 used, reset should fail.
        assert!(!pool.reset_prefix_cache());
    }

    #[test]
    fn test_free_blocks_shared_ref() {
        let mut pool = BlockPool::new(10, false, 16);
        let allocated = pool.get_new_blocks(2);

        // Simulate two references by touching.
        pool.touch(&allocated);
        // ref_cnt should be 2.
        assert_eq!(pool.block(allocated[0]).ref_cnt, 2);

        // Free once -- ref_cnt drops to 1, block stays allocated.
        pool.free_blocks(&allocated);
        assert_eq!(pool.block(allocated[0]).ref_cnt, 1);
        assert_eq!(pool.get_num_free_blocks(), 7); // no change

        // Free again -- ref_cnt drops to 0, block goes back to free list.
        pool.free_blocks(&allocated);
        assert_eq!(pool.block(allocated[0]).ref_cnt, 0);
        assert_eq!(pool.get_num_free_blocks(), 9);
    }

    #[test]
    fn test_null_block_never_freed() {
        let mut pool = BlockPool::new(5, false, 16);
        // Even if we manually set null block ref_cnt to 0, free_blocks
        // should not add it to the free list.
        let null_idx = pool.null_block_idx();
        pool.blocks[null_idx].ref_cnt = 1;
        pool.free_blocks(&[null_idx]);
        assert_eq!(pool.blocks[null_idx].ref_cnt, 0);
        // free count should remain unchanged (null block is excluded).
        assert_eq!(pool.get_num_free_blocks(), 4);
    }

    #[test]
    fn test_block_hash_to_block_map_basic() {
        let mut map = BlockHashToBlockMap::default();
        assert!(map.is_empty());

        let key = vec![1, 2, 3, 0, 0, 0, 1];
        map.insert(key.clone(), 42);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_one_block(&key), Some(42));

        // Insert another block with the same key.
        map.insert(key.clone(), 99);
        assert_eq!(map.len(), 1); // still one key
        // get_one_block returns any one of them.
        let got = map.get_one_block(&key);
        assert!(got == Some(42) || got == Some(99));

        // Pop block 42.
        assert_eq!(map.pop(&key, 42), Some(42));
        assert_eq!(map.get_one_block(&key), Some(99));

        // Pop block 99.
        assert_eq!(map.pop(&key, 99), Some(99));
        assert!(map.is_empty());
    }

    #[test]
    fn test_block_hash_to_block_map_pop_nonexistent() {
        let mut map = BlockHashToBlockMap::default();
        let key = vec![1, 2, 3];
        assert_eq!(map.pop(&key, 0), None);

        map.insert(key.clone(), 10);
        // Pop a block_id that is not under this key.
        assert_eq!(map.pop(&key, 99), None);
        // The original entry should still be there.
        assert_eq!(map.get_one_block(&key), Some(10));
    }
}
