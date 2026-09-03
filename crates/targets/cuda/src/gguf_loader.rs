// SPDX-License-Identifier: Apache-2.0
//! Cross-crate registration seam for the GGUF loader.
//!
//! `GpuWeights::from_dir` (and its `from_path` peer) auto-detect a
//! `.gguf` file in the model directory and route to a registered
//! GGUF loader. The actual loader lives in `scratchy-target-cuda` because
//! it needs the GGML dequantize kernels for norms / embeddings /
//! lm_head; this crate (`scratchy-target-cuda`) sits below
//! `scratchy-target-cuda` in the dep chain, so we use `inventory` to let
//! `scratchy-target-cuda` *register* its loader without
//! `scratchy-target-cuda` knowing about it.
//!
//! The contract:
//!
//! 1. `scratchy-target-cuda` submits a `GgufLoaderRegistration` at module
//!    init via `inventory::submit!`. The submitted fn pointer
//!    matches the signature of
//!    `crate::ggml::load_gguf_into_weights`.
//! 2. `from_dir` detects a `.gguf` file in the input directory and
//!    asks `registered_gguf_loader()` for the fn. If present, it
//!    calls through to GGUF loading; if absent, it returns a
//!    descriptive error (compile-with-`scratchy-target-cuda` is the
//!    expected fix).
//!
//! Same architecture as `scratchy-forward-compiler`'s `ScratchyArchRegistration`:
//! adding a new GGUF-style backing format means submitting a new
//! registration, not editing every consumer.
#![cfg(feature = "cuda")]

use crate::weights::GpuWeights;
use anyhow::Result;
use std::path::Path;

/// Function-pointer signature for the GGUF loader. Mirrors the
/// `unsafe fn load_gguf_into_weights` signature in
/// `crate::ggml`.
///
/// # Safety
/// Caller must hold a valid CUDA context and stream. Returned
/// `GpuWeights` retains the GGUF tensor pointers for the lifetime
/// of the model (same contract as `GgufGpuWeights::load`).
pub type GgufLoaderFn = unsafe fn(
    path: &Path,
    model_dtype: crate::DType,
    alloc: &mut crate::CachingAllocator,
    stream: crate::CUstream,
    tp_rank: usize,
    tp_world_size: usize,
) -> Result<GpuWeights>;

/// One registration per GGUF loader. Today there's exactly one:
/// `crate::ggml::load_gguf_into_weights`. The struct is
/// kept extensible (in case a future variant needs a name or a
/// priority) so adding a second backing format is a matter of
/// submitting a second `inventory::submit!` block.
pub struct GgufLoaderRegistration {
    /// Stable identifier — currently `"scratchy-target-cuda"`. Logged on
    /// load so a misregistration surfaces clearly.
    pub name: &'static str,
    /// The loader implementation.
    pub load: GgufLoaderFn,
}

inventory::collect!(GgufLoaderRegistration);

/// Look up the registered GGUF loader. Returns `None` when the
/// consuming binary didn't link `scratchy-target-cuda` (or another crate
/// that submits a `GgufLoaderRegistration`). Emit a clear error in
/// that case rather than silently degrading to safetensors.
pub fn registered_gguf_loader() -> Option<&'static GgufLoaderRegistration> {
    inventory::iter::<GgufLoaderRegistration>().next()
}
