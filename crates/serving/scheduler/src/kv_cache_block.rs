// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! KV cache block metadata and hash utilities.
//!
//! Ported from `vllm/v1/core/kv_cache_utils.py` -- specifically the
//! `KVCacheBlock` dataclass and the `BlockHash` / `BlockHashWithGroupId`
//! helper functions.

/// A `BlockHash` is a byte-string that uniquely identifies the *content* of a
/// single KV-cache block (i.e. the token IDs it covers plus any extra keys
/// such as cache salt or multi-modal hashes).
pub type BlockHash = Vec<u8>;

/// A `BlockHashWithGroupId` extends a [`BlockHash`] by appending 4 bytes that
/// encode the KV-cache group id.  This combined key is used as the lookup key
/// in the prefix-cache hash map so that blocks belonging to different cache
/// groups never collide.
pub type BlockHashWithGroupId = Vec<u8>;

// ---------------------------------------------------------------------------
// Hash helper functions
// ---------------------------------------------------------------------------

/// Pack a [`BlockHash`] and a `group_id` into a [`BlockHashWithGroupId`].
///
/// The group id is encoded as 4 big-endian bytes and appended to the block
/// hash, matching the Python implementation.
pub fn make_block_hash_with_group_id(
    block_hash: &BlockHash,
    group_id: u32,
) -> BlockHashWithGroupId {
    let mut key = block_hash.clone();
    key.extend_from_slice(&group_id.to_be_bytes());
    key
}

/// Extract the [`BlockHash`] portion from a [`BlockHashWithGroupId`].
///
/// # Panics
/// Panics if `key` has fewer than 4 bytes.
pub fn get_block_hash(key: &BlockHashWithGroupId) -> BlockHash {
    assert!(
        key.len() >= 4,
        "BlockHashWithGroupId must be at least 4 bytes"
    );
    key[..key.len() - 4].to_vec()
}

/// Extract the group id from a [`BlockHashWithGroupId`].
///
/// # Panics
/// Panics if `key` has fewer than 4 bytes.
pub fn get_group_id(key: &BlockHashWithGroupId) -> u32 {
    let len = key.len();
    assert!(len >= 4, "BlockHashWithGroupId must be at least 4 bytes");
    u32::from_be_bytes([key[len - 4], key[len - 3], key[len - 2], key[len - 1]])
}

// ---------------------------------------------------------------------------
// KVCacheBlock
// ---------------------------------------------------------------------------

/// Metadata for a single KV-cache block.
///
/// Ported from the Python `KVCacheBlock` dataclass.  In the Rust version the
/// doubly-linked-list pointers are arena indices (`Option<usize>`) rather than
/// direct references, so that all blocks can live in a flat `Vec<KVCacheBlock>`
/// without lifetime issues.
#[derive(Debug, Clone)]
pub struct KVCacheBlock {
    /// Block ID, ranging from 0 to `num_gpu_blocks - 1`.
    pub block_id: usize,

    /// Reference count -- how many active requests currently use this block.
    pub ref_cnt: u32,

    /// The combined hash key (block hash + group id).  `None` until the block
    /// is full and has been cached.
    block_hash: Option<BlockHashWithGroupId>,

    /// Index of the previous block in the free-block linked list.
    /// `None` means this block is not in the free list (or is the head
    /// sentinel).
    pub prev_free_block: Option<usize>,

    /// Index of the next block in the free-block linked list.
    /// `None` means this block is not in the free list (or is the tail
    /// sentinel).
    pub next_free_block: Option<usize>,

    /// Whether this is the null/placeholder block (block_id 0).
    /// The null block is never freed or cached.
    pub is_null: bool,
}

impl KVCacheBlock {
    /// Create a new block with the given `block_id` and default state.
    pub fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_cnt: 0,
            block_hash: None,
            prev_free_block: None,
            next_free_block: None,
            is_null: false,
        }
    }

    /// Get the block hash (read-only access).
    pub fn block_hash(&self) -> Option<&BlockHashWithGroupId> {
        self.block_hash.as_ref()
    }

    /// Set the block hash.
    ///
    /// # Panics
    /// Panics if the block already has a hash set (indicating a logic error).
    pub fn set_block_hash(&mut self, hash: BlockHashWithGroupId) {
        assert!(
            self.block_hash.is_none(),
            "Block {} already has a hash. This should not happen.",
            self.block_id
        );
        self.block_hash = Some(hash);
    }

    /// Reset (clear) the block hash.  Called when the block is evicted from
    /// the prefix cache.
    pub fn reset_hash(&mut self) {
        self.block_hash = None;
    }

    /// Returns `true` if this block is currently in the free list
    /// (i.e. has linked-list pointers set).
    pub fn is_in_free_list(&self) -> bool {
        self.prev_free_block.is_some() || self.next_free_block.is_some()
    }
}

impl std::fmt::Display for KVCacheBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KVCacheBlock(block_id={}, ref_cnt={}, has_hash={}, prev={:?}, next={:?})",
            self.block_id,
            self.ref_cnt,
            self.block_hash.is_some(),
            self.prev_free_block,
            self.next_free_block,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_block_hash_with_group_id() {
        let hash: BlockHash = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let combined = make_block_hash_with_group_id(&hash, 42);
        assert_eq!(combined.len(), 8 + 4);
        // The last 4 bytes should be the group id in big-endian.
        assert_eq!(&combined[8..], &42u32.to_be_bytes());
        // The first 8 bytes should be the original hash.
        assert_eq!(&combined[..8], &hash[..]);
    }

    #[test]
    fn test_get_block_hash() {
        let hash: BlockHash = vec![10, 20, 30];
        let combined = make_block_hash_with_group_id(&hash, 7);
        let extracted = get_block_hash(&combined);
        assert_eq!(extracted, hash);
    }

    #[test]
    fn test_get_group_id() {
        let hash: BlockHash = vec![10, 20, 30];
        let combined = make_block_hash_with_group_id(&hash, 1234);
        let gid = get_group_id(&combined);
        assert_eq!(gid, 1234);
    }

    #[test]
    fn test_roundtrip_hash_and_group_id() {
        let hash: BlockHash = vec![0xAA; 32]; // 32-byte hash like sha256
        for group_id in [0u32, 1, 255, 65535, u32::MAX] {
            let combined = make_block_hash_with_group_id(&hash, group_id);
            assert_eq!(get_block_hash(&combined), hash);
            assert_eq!(get_group_id(&combined), group_id);
        }
    }

    #[test]
    fn test_kv_cache_block_new() {
        let block = KVCacheBlock::new(5);
        assert_eq!(block.block_id, 5);
        assert_eq!(block.ref_cnt, 0);
        assert!(block.block_hash().is_none());
        assert!(block.prev_free_block.is_none());
        assert!(block.next_free_block.is_none());
        assert!(!block.is_null);
    }

    #[test]
    fn test_set_and_reset_hash() {
        let mut block = KVCacheBlock::new(0);
        assert!(block.block_hash().is_none());

        let hash = make_block_hash_with_group_id(&vec![1, 2, 3], 0);
        block.set_block_hash(hash.clone());
        assert_eq!(block.block_hash(), Some(&hash));

        block.reset_hash();
        assert!(block.block_hash().is_none());
    }

    #[test]
    #[should_panic(expected = "already has a hash")]
    fn test_set_block_hash_twice_panics() {
        let mut block = KVCacheBlock::new(0);
        let hash1 = make_block_hash_with_group_id(&vec![1, 2], 0);
        let hash2 = make_block_hash_with_group_id(&vec![3, 4], 0);
        block.set_block_hash(hash1);
        block.set_block_hash(hash2); // should panic
    }

    #[test]
    fn test_is_in_free_list() {
        let mut block = KVCacheBlock::new(0);
        assert!(!block.is_in_free_list());

        block.prev_free_block = Some(1);
        assert!(block.is_in_free_list());

        block.prev_free_block = None;
        block.next_free_block = Some(2);
        assert!(block.is_in_free_list());
    }

    #[test]
    fn test_display() {
        let block = KVCacheBlock::new(42);
        let s = format!("{}", block);
        assert!(s.contains("block_id=42"));
        assert!(s.contains("ref_cnt=0"));
    }

    #[test]
    #[should_panic(expected = "at least 4 bytes")]
    fn test_get_block_hash_too_short() {
        get_block_hash(&vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "at least 4 bytes")]
    fn test_get_group_id_too_short() {
        get_group_id(&vec![1, 2]);
    }
}
