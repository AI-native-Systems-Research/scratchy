// SPDX-License-Identifier: Apache-2.0
//! Rotary positional embedding cache — Spyre (host/KTIR) runtime half.
//!
//! Spyre is a host fp16 backend (like the GPU target): the RoPE cos/sin cache is built
//! on CPU from the model config and uploaded through the active
//! [`scratchy_tensors::WeightSource`] allocator — there is no GPU stream and no
//! kernel. The neutral [`RotaryCache`] type, the rope-scaling config structs,
//! and the stream-free `*_from_gpuweights` constructors live in
//! `scratchy-layers` (re-exported below); the macro-emitted spyre `load` body
//! calls those inherent constructors directly (`RotaryCache::new_from_gpuweights`,
//! `…::new_partial_from_gpuweights`, `…::new_proportional_from_gpuweights`),
//! exactly as the GPU target's `load` body does.

// Re-export the neutral rotary types/config (RotaryCache, *RopeScaling, …) so
// `crate::__gpu::rotary::RotaryCache` resolves.
pub use scratchy_layers::rotary::*;
