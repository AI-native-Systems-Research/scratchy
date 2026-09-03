// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral seam for the weight pre-stage ("precast") pipeline.
//!
//! Precasting is a CUDA-only load optimization: background threads pre-fault
//! mmap'd safetensors pages and stream them (casting floats on the way) into
//! pinned-host → device buffers, so `GpuWeights::take` can hand out a
//! ready device buffer without blocking. The pipeline itself (threads, pinned
//! memory, the driver DMAs) lives in `targets/cuda`; this trait is the neutral
//! handle `GpuWeights` holds so the struct names no backend type and can live
//! in `scratchy-layers`.

use crate::RawGpuMem;
use crate::dtype::DType;

/// A tensor the pre-stage pipeline has already uploaded to the device.
/// `take` adopts `gpu` into the allocator's lifetime tracker and wraps it in a
/// `GpuTensor` — no further copy or sync (the worker's per-chunk H2D is
/// blocking). The device buffer (`RawGpuMem`) is backend-neutral.
pub struct PrecastEntry {
    /// Device buffer holding the (possibly cast) tensor data.
    pub gpu: RawGpuMem,
    /// Size of valid data in bytes.
    pub size_bytes: usize,
    /// The effective dtype after casting.
    pub dtype: DType,
}

/// The pre-stage pipeline, behind a neutral trait so `GpuWeights` can hold
/// `Option<Arc<dyn PrecastPipeline>>` without naming the cuda implementation.
pub trait PrecastPipeline: Send + Sync {
    /// Claim the pre-staged device buffer for the tensor identified by its
    /// backing bytes `(mmap_base_ptr, data_offset, size_bytes)`, if one is
    /// ready. Returns `None` (and records the key as consumed) when the
    /// pipeline hasn't staged it — the caller then takes the synchronous path.
    fn take(&self, mmap_base: usize, data_offset: usize, size_bytes: usize)
    -> Option<PrecastEntry>;

    /// Signal the background workers to stop (called on `GpuWeights` drop,
    /// before the handles are joined).
    fn shutdown(&self);
}
