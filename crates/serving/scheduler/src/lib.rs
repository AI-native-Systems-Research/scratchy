// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scratchy-serving-scheduler` -- Scheduler, KV cache management, and block allocation for the
//! vLLM Rust port.
//!
//! * [`scheduler`] -- The scheduler module containing the scheduling algorithm, its KV block
//!   tracker (`SimpleBlockTracker`, behind `KVCacheManagerOps`), request queues, and output types.

pub mod scheduler;
