// SPDX-License-Identifier: Apache-2.0
//! NVIDIA CUDA backend for scratchy.
//!
//! The runtime side of the target. It holds the facts the shared passes take as input — how each
//! op's kernel treats its buffers, which steps it fuses, which result rows it samples — and the
//! step type a lowered tape is made of. Under the `nvcc` feature it also compiles and links the
//! H100 kernels (`build.rs`), exposes their launchers (`kernels`), and adds the runtime that plays
//! a tape on the GPU (`runtime`); without that feature this crate needs no CUDA toolchain to
//! build.
//!
//! A tape (`tape`) is plain static data built at expansion time by
//! `scratchy-target-cuda-compiler`: the arena's layout and the steps, each operand already a byte
//! offset or a weight index. The runtime allocates the arena once per worker and launches the
//! steps in order on one stream; it decides nothing.

/// The kernels' `extern "C"` launchers, linked from `libscratchy_cuda_kernels.a`.
#[cfg(feature = "nvcc")]
pub mod kernels;
/// CUDA's per-opcode ABI as declared data: the colour rule, fusions and sampled rows.
pub mod op_abi;
/// The rotary table's rows, built on the host from the forward's token positions.
pub mod rotary;
/// The device, its buffers, and the worker that plays a tape. Links `cudart`, so `nvcc` only.
#[cfg(feature = "nvcc")]
pub mod runtime;
/// The step a lowered tape is made of.
pub mod step;
/// One bucket's tape as static data: the arena layout and the steps.
pub mod tape;
/// One model's weight table, and the host half of its loader: the checks and the staging.
pub mod weights;
