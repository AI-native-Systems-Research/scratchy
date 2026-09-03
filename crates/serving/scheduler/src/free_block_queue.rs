// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! A doubly-linked-list queue of free KV-cache blocks.
//!
//! Ported from `FreeKVCacheBlockQueue` in `vllm/v1/core/kv_cache_utils.py`.
//!
//! The Python implementation uses direct object references for the prev/next
//! pointers.  In Rust we use arena-style indices into an external
//! `Vec<KVCacheBlock>`.  The queue itself stores only the two sentinel
//! indices and the free-block count; all actual block data lives in the
//! caller-owned arena.
//!
//! The sentinel head and tail are stored as *extra* entries appended to the
//! blocks arena by the [`BlockPool`](crate::block_pool::BlockPool) that owns
//! this queue.  Their indices are passed at construction time.

use crate::kv_cache_block::KVCacheBlock;

/// A doubly-linked-list queue of free KV-cache blocks, implemented with
/// arena indices.
///
/// The queue maintains LRU eviction order: newly freed blocks are appended
/// at the tail, and allocation pops from the head.  Removing a block from
/// the middle (e.g. when a cached block is touched) is O(1).
///
/// # Sentinel nodes
///
/// The queue uses two sentinel (fake) blocks whose indices are
/// `head_sentinel_idx` and `tail_sentinel_idx`.  These sentinels are never
/// popped or returned to the caller; they exist solely to eliminate
/// special-case branches for empty-list / single-element operations.
#[derive(Debug)]
pub struct FreeKVCacheBlockQueue {
    /// Number of *real* free blocks currently in the queue.
    num_free_blocks: usize,
    /// Arena index of the head sentinel block.
    head_sentinel_idx: usize,
    /// Arena index of the tail sentinel block.
    tail_sentinel_idx: usize,
}

impl FreeKVCacheBlockQueue {
    /// Create a new free-block queue and link together the blocks whose
    /// indices are `0..num_blocks` in the given arena.
    ///
    /// The sentinel blocks must already exist at `head_sentinel_idx` and
    /// `tail_sentinel_idx` in the arena.
    ///
    /// The blocks `0..num_blocks` are linked in ascending index order and
    /// placed between the two sentinels.
    pub fn new(
        blocks: &mut [KVCacheBlock],
        num_blocks: usize,
        head_sentinel_idx: usize,
        tail_sentinel_idx: usize,
    ) -> Self {
        // Link consecutive real blocks.
        #[allow(clippy::needless_range_loop)]
        for i in 0..num_blocks {
            if i > 0 {
                blocks[i].prev_free_block = Some(i - 1);
            }
            if i + 1 < num_blocks {
                blocks[i].next_free_block = Some(i + 1);
            }
        }

        // Wire up the sentinels.
        if num_blocks > 0 {
            blocks[head_sentinel_idx].next_free_block = Some(0);
            blocks[0].prev_free_block = Some(head_sentinel_idx);

            blocks[tail_sentinel_idx].prev_free_block = Some(num_blocks - 1);
            blocks[num_blocks - 1].next_free_block = Some(tail_sentinel_idx);
        } else {
            // Empty list -- sentinels point at each other.
            blocks[head_sentinel_idx].next_free_block = Some(tail_sentinel_idx);
            blocks[tail_sentinel_idx].prev_free_block = Some(head_sentinel_idx);
        }

        Self {
            num_free_blocks: num_blocks,
            head_sentinel_idx,
            tail_sentinel_idx,
        }
    }

    /// The number of free blocks currently in the queue.
    pub fn num_free_blocks(&self) -> usize {
        self.num_free_blocks
    }

    /// Pop the first (head) free block from the queue.
    ///
    /// Returns the arena index of the popped block.
    ///
    /// # Panics
    /// Panics if the queue is empty.
    pub fn popleft(&mut self, blocks: &mut [KVCacheBlock]) -> usize {
        let first_idx = blocks[self.head_sentinel_idx]
            .next_free_block
            .expect("head sentinel must have a next pointer");

        assert_ne!(
            first_idx, self.tail_sentinel_idx,
            "No free blocks available (num_free_blocks={})",
            self.num_free_blocks
        );

        let second_idx = blocks[first_idx]
            .next_free_block
            .expect("free block must have next_free_block set");

        // head_sentinel -> second (skip first)
        blocks[self.head_sentinel_idx].next_free_block = Some(second_idx);
        blocks[second_idx].prev_free_block = Some(self.head_sentinel_idx);

        // Detach the popped block.
        blocks[first_idx].prev_free_block = None;
        blocks[first_idx].next_free_block = None;

        self.num_free_blocks -= 1;
        first_idx
    }

    /// Pop the first `n` free blocks from the queue.
    ///
    /// Returns a `Vec` of arena indices.
    ///
    /// # Panics
    /// Panics if `n > self.num_free_blocks()`.
    pub fn popleft_n(&mut self, blocks: &mut [KVCacheBlock], n: usize) -> Vec<usize> {
        if n == 0 {
            return Vec::new();
        }
        assert!(
            self.num_free_blocks >= n,
            "Cannot pop {} blocks; only {} free",
            n,
            self.num_free_blocks,
        );

        self.num_free_blocks -= n;

        let mut result = Vec::with_capacity(n);
        let mut curr_idx = blocks[self.head_sentinel_idx]
            .next_free_block
            .expect("head sentinel next must exist");

        for _ in 0..n {
            result.push(curr_idx);
            let next_idx = blocks[curr_idx]
                .next_free_block
                .expect("free block must have next_free_block");
            // Detach the popped block.
            blocks[curr_idx].prev_free_block = None;
            blocks[curr_idx].next_free_block = None;
            curr_idx = next_idx;
        }

        // `curr_idx` now points to either the next real block or the tail
        // sentinel. Re-link head_sentinel -> curr_idx.
        blocks[self.head_sentinel_idx].next_free_block = Some(curr_idx);
        blocks[curr_idx].prev_free_block = Some(self.head_sentinel_idx);

        result
    }

    /// Remove a specific block from the free list.
    ///
    /// The block must currently be in the free list (i.e. have both prev and
    /// next pointers set).
    ///
    /// # Panics
    /// Panics if the block is not in the free list.
    pub fn remove(&mut self, blocks: &mut [KVCacheBlock], block_idx: usize) {
        let prev_idx = blocks[block_idx]
            .prev_free_block
            .expect("remove() called on a block not in the free list (no prev)");
        let next_idx = blocks[block_idx]
            .next_free_block
            .expect("remove() called on a block not in the free list (no next)");

        blocks[prev_idx].next_free_block = Some(next_idx);
        blocks[next_idx].prev_free_block = Some(prev_idx);

        blocks[block_idx].prev_free_block = None;
        blocks[block_idx].next_free_block = None;

        self.num_free_blocks -= 1;
    }

    /// Append a single block to the tail of the free list.
    pub fn append(&mut self, blocks: &mut [KVCacheBlock], block_idx: usize) {
        let last_idx = blocks[self.tail_sentinel_idx]
            .prev_free_block
            .expect("tail sentinel must have a prev pointer");

        // last -> block -> tail_sentinel
        blocks[last_idx].next_free_block = Some(block_idx);
        blocks[block_idx].prev_free_block = Some(last_idx);
        blocks[block_idx].next_free_block = Some(self.tail_sentinel_idx);
        blocks[self.tail_sentinel_idx].prev_free_block = Some(block_idx);

        self.num_free_blocks += 1;
    }

    /// Append multiple blocks to the tail of the free list.
    pub fn append_n(&mut self, blocks: &mut [KVCacheBlock], block_indices: &[usize]) {
        if block_indices.is_empty() {
            return;
        }

        let mut last_idx = blocks[self.tail_sentinel_idx]
            .prev_free_block
            .expect("tail sentinel must have a prev pointer");

        for &idx in block_indices {
            blocks[idx].prev_free_block = Some(last_idx);
            blocks[last_idx].next_free_block = Some(idx);
            last_idx = idx;
        }

        // Connect the last appended block to the tail sentinel.
        blocks[last_idx].next_free_block = Some(self.tail_sentinel_idx);
        blocks[self.tail_sentinel_idx].prev_free_block = Some(last_idx);

        self.num_free_blocks += block_indices.len();
    }

    /// Collect all free block indices in order (head to tail).
    /// Mainly used for testing.
    pub fn get_all_free_blocks(&self, blocks: &[KVCacheBlock]) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.num_free_blocks);
        let mut curr_idx = blocks[self.head_sentinel_idx]
            .next_free_block
            .expect("head sentinel next must exist");

        while curr_idx != self.tail_sentinel_idx {
            result.push(curr_idx);
            curr_idx = blocks[curr_idx]
                .next_free_block
                .expect("free block must have next_free_block");
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache_block::KVCacheBlock;

    /// Helper: create an arena with `n` real blocks plus two sentinel blocks.
    /// Returns (arena, head_sentinel_idx, tail_sentinel_idx).
    fn make_arena(n: usize) -> (Vec<KVCacheBlock>, usize, usize) {
        let head_idx = n;
        let tail_idx = n + 1;
        let mut arena: Vec<KVCacheBlock> = (0..n).map(KVCacheBlock::new).collect();
        // Sentinel blocks with special IDs.
        arena.push(KVCacheBlock::new(usize::MAX - 1)); // head sentinel
        arena.push(KVCacheBlock::new(usize::MAX)); // tail sentinel
        (arena, head_idx, tail_idx)
    }

    #[test]
    fn test_new_empty() {
        let (mut arena, head, tail) = make_arena(0);
        let q = FreeKVCacheBlockQueue::new(&mut arena, 0, head, tail);
        assert_eq!(q.num_free_blocks(), 0);
        assert!(q.get_all_free_blocks(&arena).is_empty());
    }

    #[test]
    fn test_new_with_blocks() {
        let (mut arena, head, tail) = make_arena(5);
        let q = FreeKVCacheBlockQueue::new(&mut arena, 5, head, tail);
        assert_eq!(q.num_free_blocks(), 5);
        assert_eq!(q.get_all_free_blocks(&arena), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_popleft() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);

        let idx = q.popleft(&mut arena);
        assert_eq!(idx, 0);
        assert_eq!(q.num_free_blocks(), 2);
        assert_eq!(q.get_all_free_blocks(&arena), vec![1, 2]);

        // The popped block should not be in the free list.
        assert!(arena[0].prev_free_block.is_none());
        assert!(arena[0].next_free_block.is_none());
    }

    #[test]
    fn test_popleft_n() {
        let (mut arena, head, tail) = make_arena(5);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 5, head, tail);

        let popped = q.popleft_n(&mut arena, 3);
        assert_eq!(popped, vec![0, 1, 2]);
        assert_eq!(q.num_free_blocks(), 2);
        assert_eq!(q.get_all_free_blocks(&arena), vec![3, 4]);

        // Popped blocks should be detached.
        for &idx in &popped {
            assert!(arena[idx].prev_free_block.is_none());
            assert!(arena[idx].next_free_block.is_none());
        }
    }

    #[test]
    fn test_popleft_n_all() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);

        let popped = q.popleft_n(&mut arena, 3);
        assert_eq!(popped, vec![0, 1, 2]);
        assert_eq!(q.num_free_blocks(), 0);
        assert!(q.get_all_free_blocks(&arena).is_empty());
    }

    #[test]
    fn test_popleft_n_zero() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);
        let popped = q.popleft_n(&mut arena, 0);
        assert!(popped.is_empty());
        assert_eq!(q.num_free_blocks(), 3);
    }

    #[test]
    #[should_panic(expected = "Cannot pop")]
    fn test_popleft_n_too_many() {
        let (mut arena, head, tail) = make_arena(2);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 2, head, tail);
        q.popleft_n(&mut arena, 5);
    }

    #[test]
    fn test_remove_middle() {
        let (mut arena, head, tail) = make_arena(5);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 5, head, tail);

        // Remove block 2 (middle).
        q.remove(&mut arena, 2);
        assert_eq!(q.num_free_blocks(), 4);
        assert_eq!(q.get_all_free_blocks(&arena), vec![0, 1, 3, 4]);
    }

    #[test]
    fn test_remove_head() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);

        q.remove(&mut arena, 0);
        assert_eq!(q.num_free_blocks(), 2);
        assert_eq!(q.get_all_free_blocks(&arena), vec![1, 2]);
    }

    #[test]
    fn test_remove_tail() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);

        q.remove(&mut arena, 2);
        assert_eq!(q.num_free_blocks(), 2);
        assert_eq!(q.get_all_free_blocks(&arena), vec![0, 1]);
    }

    #[test]
    fn test_append() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);

        // Pop all, then append back in a different order.
        let _ = q.popleft_n(&mut arena, 3);
        assert_eq!(q.num_free_blocks(), 0);

        q.append(&mut arena, 2);
        q.append(&mut arena, 0);
        q.append(&mut arena, 1);
        assert_eq!(q.num_free_blocks(), 3);
        assert_eq!(q.get_all_free_blocks(&arena), vec![2, 0, 1]);
    }

    #[test]
    fn test_append_n() {
        let (mut arena, head, tail) = make_arena(5);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 5, head, tail);

        let _ = q.popleft_n(&mut arena, 5);
        assert_eq!(q.num_free_blocks(), 0);

        q.append_n(&mut arena, &[4, 3, 2]);
        assert_eq!(q.num_free_blocks(), 3);
        assert_eq!(q.get_all_free_blocks(&arena), vec![4, 3, 2]);

        q.append_n(&mut arena, &[1, 0]);
        assert_eq!(q.num_free_blocks(), 5);
        assert_eq!(q.get_all_free_blocks(&arena), vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_append_n_empty() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);
        q.append_n(&mut arena, &[]);
        assert_eq!(q.num_free_blocks(), 3);
    }

    #[test]
    fn test_pop_remove_append_cycle() {
        // Simulate a realistic allocation cycle:
        // 1. Pop some blocks (allocate)
        // 2. Touch a cached block (remove from free list)
        // 3. Free blocks (append back)
        let (mut arena, head, tail) = make_arena(5);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 5, head, tail);

        // Allocate blocks 0 and 1.
        let allocated = q.popleft_n(&mut arena, 2);
        assert_eq!(allocated, vec![0, 1]);
        assert_eq!(q.num_free_blocks(), 3);

        // Touch block 3 (it is still free, remove it).
        q.remove(&mut arena, 3);
        assert_eq!(q.num_free_blocks(), 2);
        assert_eq!(q.get_all_free_blocks(&arena), vec![2, 4]);

        // Free blocks 1 and 0 (in reverse order for LRU).
        q.append_n(&mut arena, &[1, 0]);
        assert_eq!(q.num_free_blocks(), 4);
        assert_eq!(q.get_all_free_blocks(&arena), vec![2, 4, 1, 0]);
    }

    #[test]
    #[should_panic(expected = "No free blocks")]
    fn test_popleft_empty_panics() {
        let (mut arena, head, tail) = make_arena(0);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 0, head, tail);
        q.popleft(&mut arena);
    }

    #[test]
    #[should_panic(expected = "not in the free list")]
    fn test_remove_detached_block_panics() {
        let (mut arena, head, tail) = make_arena(3);
        let mut q = FreeKVCacheBlockQueue::new(&mut arena, 3, head, tail);
        let idx = q.popleft(&mut arena);
        // `idx` is no longer in the free list, removing it should panic.
        q.remove(&mut arena, idx);
    }
}
