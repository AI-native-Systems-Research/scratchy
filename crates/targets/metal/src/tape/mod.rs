// SPDX-License-Identifier: Apache-2.0
//! The metal tape-construction layer — the pure (objc-free) half of this
//! crate. The `#[forward]` macro depends on this crate under `-Fmetal`
//! (macOS-only builds; the linux CI graph never activates it) and RUNS
//! the same command construction the runtime uses, emitting the results
//! as static literals: one implementation, called by the macro at
//! expansion, with the runtime playing the emitted statics.
pub mod constants;
pub mod continuation_witness;
pub mod ids;
pub mod kernel_bindings;
pub mod kernel_constants;
pub mod kernel_identity;
pub mod lowered;
pub mod lowering;
pub mod model_consts;
pub mod quantized;
pub mod targets;
