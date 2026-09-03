// SPDX-License-Identifier: Apache-2.0
//! Re-export of the backend-neutral [`GpuWeights`](scratchy_layers::weights::GpuWeights)
//! plus the CUDA surface (extension trait + precast pipeline + constructors) and
//! the GGUF/quant loaders.
//!
//! The struct + host loader are generic over `A: DeviceAllocator` and live in
//! `scratchy-layers`; the defaulted alias preserves the historical
//! `scratchy_target_cuda::weights::GpuWeights` (with `A = BackendAllocator`)
//! path. The CUDA driver surface rides [`CudaWeightsExt`] / `weights_cuda`.

/// Backend-neutral model weights, with the allocator defaulted to this crate's
/// [`BackendAllocator`](crate::BackendAllocator).
pub type GpuWeights<A = crate::BackendAllocator> = scratchy_layers::weights::GpuWeights<A>;

pub use scratchy_layers::weights::UploadSrc;

#[cfg(feature = "cuda")]
pub use crate::weights_cuda::{CudaWeightsExt, from_gguf_file, from_path};

// Quantized weight loaders (Marlin / BNB4 / FP8 block) live in the sibling
// `weights_quant` module; surface them here so `crate::weights::<loader>`
// resolves the merged GPU-weights API.
#[cfg(feature = "cuda")]
pub use crate::weights_quant::*;
