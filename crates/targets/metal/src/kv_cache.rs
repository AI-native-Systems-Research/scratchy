// SPDX-License-Identifier: Apache-2.0
//! Re-export of the backend-neutral [`KvCachePool`](scratchy_layers::kv_cache::KvCachePool).
//!
//! The pool storage + layout is generic over `M: PoolMemory` and lives in
//! `scratchy-layers`; the defaulted alias preserves the
//! `scratchy_target_metal::kv_cache::KvCachePool` (with `M = PoolMem` =
//! [`MetalMem`](crate::MetalMem)) path the worker + macro emissions name. Every
//! method the metal path uses (`new_metal_chunked`, `empty_for_vision`,
//! `k_layer_mem` / `v_layer_mem`, `k_chunk_table_mem` / `v_chunk_table_mem`) is
//! a neutral inherent method on the layers pool — no metal extension trait is
//! needed (unlike the cuda target, whose `driver`-DMA methods ride a
//! `CudaKvCacheExt`).
pub type KvCachePool<M = crate::PoolMem> = scratchy_layers::kv_cache::KvCachePool<M>;
