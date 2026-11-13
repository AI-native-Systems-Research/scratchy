// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The spyre worker: `SpyreWorker` implements the serving `Worker` trait over the spyre
//! target, on a card (`spyre-hw`) or the emulator (`spyre-emu`), and `SpyreWorkerFactory`
//! registers it in the `WorkerFactory` registry. A binary that serves on spyre links this
//! crate; the serving crates never name it.

#[cfg(feature = "spyre-hw")]
mod card;
#[cfg(all(feature = "spyre-emu", not(feature = "spyre-hw")))]
mod emu;
#[cfg(any(feature = "spyre-emu", feature = "spyre-hw"))]
pub mod spyre_exec;
#[cfg(any(feature = "spyre-emu", feature = "spyre-hw"))]
pub mod spyre_forward;
#[cfg(any(feature = "spyre-emu", feature = "spyre-hw"))]
pub mod spyre_load;
#[cfg(any(feature = "spyre-emu", feature = "spyre-hw"))]
pub mod spyre_types;
#[cfg(any(feature = "spyre-emu", feature = "spyre-hw"))]
pub mod spyre_worker;
