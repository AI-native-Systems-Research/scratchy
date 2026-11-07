// SPDX-License-Identifier: Apache-2.0
//! CUDA'S COMPILE-TIME LOWERING.
//!
//! ⭐ TARGET CODEGEN, IN THE TARGET'S DIRECTORY — the counterpart of `scratchy-target-metal-compiler`.
//! [`canonical::plan_canonical`] runs the shared passes (sampled rows, tape, folds, colours) under
//! CUDA's declared tables; [`canonical::lower_canonical`] adds CUDA's opcode lowering
//! ([`steps_from_tape`]) on top. [`emit::emit_model`] is the `#[forward]` macro's entry: it
//! lowers every canonical of one model and emits its `static` tapes.

/// One decode canonical, planned and lowered.
pub mod canonical;
/// One model's tapes, lowered per bucket and emitted as `static` tokens: the macro's entry.
pub mod emit;
/// CUDA's opcode lowering: one tape step to one [`scratchy_target_cuda::step::CudaStep`].
pub mod steps_from_tape;
