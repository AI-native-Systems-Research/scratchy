// SPDX-License-Identifier: Apache-2.0
//! NVIDIA CUDA backend for scratchy.
//!
//! The runtime side of the target. So far it holds the facts the shared passes take as input —
//! how each op's kernel treats its buffers, which steps it fuses, which result rows it samples —
//! and the step type a lowered tape is made of. Kernels, the device and the tape player come with
//! later issues; until then this crate needs no CUDA toolchain to build.

/// CUDA's per-opcode ABI as declared data: the colour rule, fusions and sampled rows.
pub mod op_abi;
/// The step a lowered tape is made of.
pub mod step;
