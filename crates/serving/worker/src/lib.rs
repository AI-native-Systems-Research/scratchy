// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Worker/process management and distributed coordination.
//!
//! This crate provides:
//!
//! * [`worker`] -- The `Worker` trait defining the device-level execution interface.
//!   Port of `vllm/v1/worker/worker_base.py`.
//! * [`uniproc`] -- `UniProcExecutor`: single-process executor wrapping one worker.
//!   Port of `vllm/v1/executor/uniproc_executor.py`.
//! * [`threadpool`] -- `ThreadPoolExecutor`: thread-pool executor for multi-GPU TP.
//! * [`parallel`] -- Distributed parallel state types (TP/PP groups, rank management).
//! * [`error`] -- Executor-specific error types.

pub mod error;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu_worker;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu_worker_base;
pub mod input_batch;
#[cfg(feature = "metal")]
pub mod metal_info;
#[cfg(feature = "nccl")]
pub mod multinode;
pub mod parallel;
#[cfg(any(feature = "spyre", feature = "spyre-hw"))]
pub mod spyre_exec;
#[cfg(any(feature = "spyre", feature = "spyre-hw"))]
pub mod spyre_forward;
#[cfg(any(feature = "spyre", feature = "spyre-hw"))]
pub mod spyre_load;
#[cfg(any(feature = "spyre", feature = "spyre-hw"))]
pub mod spyre_pool;
#[cfg(any(feature = "spyre", feature = "spyre-hw"))]
pub mod spyre_types;
#[cfg(any(feature = "spyre", feature = "spyre-hw"))]
pub mod spyre_worker;
pub mod threadpool;
pub mod uniproc;
pub mod worker;
pub mod worker_factory;
