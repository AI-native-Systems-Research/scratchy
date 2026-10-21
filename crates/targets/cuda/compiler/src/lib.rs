// SPDX-License-Identifier: Apache-2.0
//! CUDA'S COMPILE-TIME LOWERING.
//!
//! ⭐ TARGET CODEGEN, IN THE TARGET'S DIRECTORY — the counterpart of `scratchy-target-metal-compiler`.
//! [`canonical::plan_canonical`] runs the shared passes (sampled rows, tape, folds, colours) under
//! CUDA's declared tables; [`canonical::lower_canonical`] adds CUDA's opcode lowering
//! ([`steps_from_tape`]) on top. The `#[forward]` macro does not call either yet.

/// One decode canonical, planned and lowered.
pub mod canonical;
/// CUDA's opcode lowering: one tape step to one [`scratchy_target_cuda::step::CudaStep`].
pub mod steps_from_tape;
