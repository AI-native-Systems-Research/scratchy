// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine core loop and request lifecycle for vLLM.
//!
//! This crate provides:
//!
//! * [`engine_core`] -- The `EngineCore` struct that orchestrates the scheduler
//!   and executor. This is the inner loop of the vLLM engine.
//! * [`executor`] -- The `Executor` trait defining the interface to GPU workers.
//!   During the transition, Python workers implement this via PyO3.
//! * [`error`] -- Engine-specific error types.

pub mod core_client;
pub mod engine_core;
pub mod error;
pub mod executor;
pub mod gpu_budget;
pub mod input_batch;
pub mod spec_decode;
pub mod worker;
pub mod worker_factory;
