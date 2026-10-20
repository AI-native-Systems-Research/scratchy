// SPDX-License-Identifier: Apache-2.0
//! NVIDIA CUDA backend for scratchy.
//!
//! The runtime side of the target. It holds the facts the shared passes take as input — how each
//! op's kernel treats its buffers, which steps it fuses, which result rows it samples — and the
//! step type a lowered tape is made of. Under the `nvcc` feature it also compiles and links the
//! H100 kernels (`build.rs`) and exposes their launchers (`kernels`); without that feature this
//! crate needs no CUDA toolchain to build. The device and the tape player come with later issues.

/// The kernels' `extern "C"` launchers, linked from `libscratchy_cuda_kernels.a`.
#[cfg(feature = "nvcc")]
pub mod kernels;
/// CUDA's per-opcode ABI as declared data: the colour rule, fusions and sampled rows.
pub mod op_abi;
/// The step a lowered tape is made of.
pub mod step;
