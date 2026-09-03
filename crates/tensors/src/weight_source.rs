// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral, format-agnostic view of a weight store.
//!
//! The minimal vocabulary that per-format quantized `load()` paths need from a
//! weight store (`take`/`take_cpu`/`alloc_packed_from_host`/…). Implemented by
//! the backend crates' `GpuWeights`, it lets per-format loader types (e.g. the
//! MLX-affine / NVFP4 quant linears) live in a backend crate *below* the one
//! that owns `GpuWeights` — both target crates already depend on
//! `scratchy-tensors`, so this trait is the one name both sides can reach.
//!
//! Deliberately FROZEN to this method set. It must stay object-safe and must
//! NOT grow allocator / precast / streaming methods (`CUstream`, `RawGpuMem`,
//! pinned-host DMA) — those would drag a backend back into this neutral crate.
//! A loader needing more belongs behind a separate, backend-local trait.

use crate::{DType, GpuTensor};
use anyhow::Result;

/// Neutral weight-store accessor consumed by quantized loader types.
pub trait WeightSource {
    /// Take a tensor by name, uploading to device in the store's target dtype.
    fn take(&mut self, name: &str) -> Result<GpuTensor>;

    /// Take a tensor by name, preserving its on-disk dtype (no cast).
    fn take_keep_dtype(&mut self, name: &str) -> Result<GpuTensor>;

    /// Take a tensor's raw bytes to host: `(bytes, shape, dtype)`.
    fn take_cpu(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, DType)>;

    /// Take a tensor to host as `f32` (casting from its on-disk dtype).
    fn take_to_cpu_f32(&mut self, name: &str) -> Result<Vec<f32>>;

    /// Whether a tensor with `name` is present in the store.
    fn contains(&self, name: &str) -> bool;

    /// Allocate a device tensor from packed host bytes with the given shape/dtype.
    fn alloc_packed_from_host(
        &mut self,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<GpuTensor>;
}
