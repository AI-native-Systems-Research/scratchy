// SPDX-License-Identifier: Apache-2.0
// The rotary loader mirrors cuda/metal's multi-arg `new_*` signatures.
#![allow(clippy::too_many_arguments)]
//! Rotary positional embedding cache — Spyre (host/KTIR) runtime half.
//!
//! Spyre is a host fp16 backend (like metal): the RoPE cos/sin cache is built
//! on CPU from the model config and uploaded through the active
//! [`scratchy_tensors::WeightSource`] allocator — there is no GPU stream and no
//! kernel. The neutral [`RotaryCache`] type, the rope-scaling config structs,
//! and the stream-free `*_from_gpuweights` constructors live in
//! `scratchy-layers` (re-exported below); the macro-emitted spyre `load` body
//! calls those inherent constructors directly (`RotaryCache::new_from_gpuweights`,
//! `…::new_partial_from_gpuweights`, `…::new_proportional_from_gpuweights`),
//! exactly as the metal `load` body does.
//!
//! The macro additionally brings [`CudaRotaryExt`] into scope
//! (`use crate::__gpu::rotary::CudaRotaryExt as _;`) so the cuda-shaped load
//! body's path-syntax `RotaryCache::new_*_from_stream(...)` calls resolve under
//! every target. Spyre keeps the trait name (the macro imports it by that name)
//! and mirrors the cuda method set, but the stream-based constructors are
//! genuinely stream/device-bound — Spyre has no CUDA stream — so they
//! `anyhow::bail!` with a recoverable error. The working host path is the
//! `*_from_gpuweights` inherent constructors; a Spyre `load` never reaches the
//! `*_from_stream` arms (it emits the GpuWeights-based ones), but they must
//! still type-check so the shared cuda-shaped body compiles.

// Re-export the neutral rotary types/config (RotaryCache, *RopeScaling, …) so
// `crate::__gpu::rotary::RotaryCache` resolves. The glob also brings them into
// this module's scope for the trait impl below — no separate (shadowing) `use`.
pub use scratchy_layers::rotary::*;

use crate::dtype::DType;
use anyhow::Result;

/// Spyre [`RotaryCache`] stream-based constructors. Extension trait over the
/// neutral type, mirroring `scratchy_target_cuda::rotary::CudaRotaryExt` so the
/// macro-emitted load body's `use crate::__gpu::rotary::CudaRotaryExt as _;`
/// resolves under `-Fspyre` (bring it into scope with that import).
///
/// Spyre is a host backend with no CUDA stream, so the `*_from_stream`
/// constructors here are not the working load path — the macro's spyre body
/// builds the cache through the neutral `RotaryCache::new_*_from_gpuweights`
/// inherent constructors (allocator-generic host trig + `WeightSource` upload).
/// These trait methods exist for signature parity with the cuda surface and
/// return a recoverable [`anyhow::Error`] if ever called on Spyre.
pub trait CudaRotaryExt: Sized {
    /// Build the cos/sin cache (cuda: on GPU via the device's default stream).
    /// Not the Spyre host path — use `RotaryCache::new_from_gpuweights`.
    ///
    /// # Safety
    /// Mirrors the cuda signature; the Spyre body performs no unsafe work.
    unsafe fn new_from_stream(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: crate::LoadStream,
    ) -> Result<Self>;

    /// Phi-3 / Phi-3.5 LongRoPE (su-scaling) variant of `new_from_stream`.
    /// Not the Spyre host path.
    ///
    /// # Safety
    /// Mirrors the cuda signature; the Spyre body performs no unsafe work.
    unsafe fn new_longrope_from_stream(
        head_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: crate::LoadStream,
    ) -> Result<Self>;

    /// Phi-4-mini variant: partial rotary (`rotary_dim < head_dim`) + LongRoPE.
    /// Not the Spyre host path.
    ///
    /// # Safety
    /// Mirrors the cuda signature; the Spyre body performs no unsafe work.
    unsafe fn new_partial_longrope_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        max_model_len: usize,
        rope_theta: f64,
        longrope: &LongRopeScaling,
        dtype: DType,
        stream: crate::LoadStream,
    ) -> Result<Self>;

    /// Build a YaRN NTK-by-parts RoPE cos/sin cache for DeepSeek-V2 MLA.
    /// Not the Spyre host path.
    ///
    /// # Safety
    /// Mirrors the cuda signature; the Spyre body performs no unsafe work.
    unsafe fn new_yarn_from_stream(
        rope_head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        yarn: &YarnRopeScaling,
        dtype: DType,
        stream: crate::LoadStream,
    ) -> Result<Self>;

    /// Partial-rotary variant of `new_from_stream` (`rotary_dim < head_dim`).
    /// Not the Spyre host path — use `RotaryCache::new_partial_from_gpuweights`.
    ///
    /// # Safety
    /// Mirrors the cuda signature; the Spyre body performs no unsafe work.
    unsafe fn new_partial_from_stream(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        stream: crate::LoadStream,
    ) -> Result<Self>;
}

impl CudaRotaryExt for RotaryCache {
    unsafe fn new_from_stream(
        _head_dim: usize,
        _max_pos: usize,
        _rope_theta: f64,
        _llama3_scaling: Option<&Llama3RopeScaling>,
        _dtype: DType,
        _stream: crate::LoadStream,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: RotaryCache::new_from_stream load not yet implemented \
             (Spyre has no CUDA stream — use RotaryCache::new_from_gpuweights)"
        )
    }

    unsafe fn new_longrope_from_stream(
        _head_dim: usize,
        _max_pos: usize,
        _max_model_len: usize,
        _rope_theta: f64,
        _longrope: &LongRopeScaling,
        _dtype: DType,
        _stream: crate::LoadStream,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: RotaryCache::new_longrope_from_stream load not yet implemented"
        )
    }

    unsafe fn new_partial_longrope_from_stream(
        _head_dim: usize,
        _rotary_dim: usize,
        _max_pos: usize,
        _max_model_len: usize,
        _rope_theta: f64,
        _longrope: &LongRopeScaling,
        _dtype: DType,
        _stream: crate::LoadStream,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: RotaryCache::new_partial_longrope_from_stream load not yet implemented"
        )
    }

    unsafe fn new_yarn_from_stream(
        _rope_head_dim: usize,
        _max_pos: usize,
        _rope_theta: f64,
        _yarn: &YarnRopeScaling,
        _dtype: DType,
        _stream: crate::LoadStream,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: RotaryCache::new_yarn_from_stream load not yet implemented"
        )
    }

    unsafe fn new_partial_from_stream(
        _head_dim: usize,
        _rotary_dim: usize,
        _max_pos: usize,
        _rope_theta: f64,
        _llama3_scaling: Option<&Llama3RopeScaling>,
        _dtype: DType,
        _stream: crate::LoadStream,
    ) -> Result<Self> {
        anyhow::bail!(
            "scratchy-target-spyre: RotaryCache::new_partial_from_stream load not yet implemented \
             (Spyre has no CUDA stream — use RotaryCache::new_partial_from_gpuweights)"
        )
    }
}
