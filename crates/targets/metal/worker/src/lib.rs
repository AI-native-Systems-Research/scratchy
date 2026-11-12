// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The metal worker: `GpuWorker` implements the serving `Worker` trait over the metal target,
//! and `MetalWorkerFactory` registers it in the `WorkerFactory` registry. A binary that serves
//! on metal links this crate (`scr`'s `metal` feature); the serving crates never name it.

pub mod gpu_worker;
pub mod gpu_worker_base;
pub mod metal_info;
