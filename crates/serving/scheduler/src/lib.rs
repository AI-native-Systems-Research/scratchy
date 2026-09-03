// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scratchy-serving-scheduler` -- Scheduler, KV cache management, and block allocation for the
//! vLLM Rust port.
//!
//! This crate provides the core scheduling and KV cache management
//! infrastructure:
//!
//! * [`kv_cache_block`] -- KV cache block metadata and hash utilities.
//! * [`free_block_queue`] -- Doubly-linked free-block queue.
//! * [`block_pool`] -- Block pool (arena + free list + prefix cache map).
//! * [`kv_cache_manager`] -- High-level KV cache manager used by the scheduler.
//! * [`scheduler`] -- The scheduler module containing the scheduling algorithm,
//!   request queues, and output types.

pub mod block_pool;
pub mod free_block_queue;
pub mod kv_cache_block;
pub mod kv_cache_manager;
pub mod scheduler;

// ---- Convenience re-exports ------------------------------------------------

pub use block_pool::BlockPool;
pub use kv_cache_block::{
    BlockHash, BlockHashWithGroupId, KVCacheBlock, get_block_hash, get_group_id,
    make_block_hash_with_group_id,
};
pub use kv_cache_manager::KVCacheManager;
