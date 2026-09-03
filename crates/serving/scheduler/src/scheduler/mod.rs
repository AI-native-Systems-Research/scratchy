// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Scheduler module for the vLLM Rust engine core.
//!
//! This module contains the scheduling logic ported from
//! `vllm/v1/core/sched/` in the Python codebase. It provides:
//!
//! * [`interface`] -- The `SchedulerInterface` trait and `PauseState` enum.
//! * [`output`] -- Scheduler output types (`SchedulerOutput`, `NewRequestData`,
//!   `CachedRequestData`).
//! * [`request_queue`] -- Request queue implementations (FCFS and Priority).
//! * [`core`] -- The main `Scheduler` struct and its scheduling algorithm.

pub mod core;
pub mod interface;
pub mod output;
pub mod request_queue;

// Re-export the key types for convenience.
pub use self::core::Scheduler;
pub use interface::{PauseState, SchedulerInterface};
pub use output::{CachedRequestData, NewRequestData, SchedulerOutput};
pub use request_queue::{
    FCFSRequestQueue, PriorityRequestQueue, RequestQueue, SchedulingPolicy, create_request_queue,
};
