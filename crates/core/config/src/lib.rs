// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! # scratchy-core-config
//!
//! Configuration types for the vLLM Rust port.
//!
//! This crate provides Rust equivalents of the Python configuration dataclasses
//! found in `vllm/config/` and `vllm/v1/kv_cache_interface.py`.  All types
//! derive `Debug`, `Clone`, `Serialize`, and `Deserialize` so they can be
//! round-tripped through JSON (or MessagePack via `rmp-serde`) when crossing
//! the Python/Rust boundary through PyO3.

pub mod cache;
pub mod compilation;
pub mod scheduler;
pub mod spans;

// Re-export top-level types for convenience.
pub use cache::{
    CacheConfig, CacheDType, HybridKvLayout, KVCacheConfig, KVCacheGroupSpec, KVCacheSpec,
    KVCacheSpecType, KVCacheTensor, LayerKvGeometry, MambaCacheMode, PrefixCachingHashAlgo,
    compute_hybrid_kv_layout,
};
pub use compilation::{CudaGraphConfig, CudaGraphMode};
pub use scheduler::{RunnerType, SchedulerConfig, SchedulerPolicy};
pub use spans::SpansConfig;
