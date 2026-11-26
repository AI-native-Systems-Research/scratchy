// SPDX-License-Identifier: Apache-2.0
//! THE CUDA MODEL REGISTRY — every model this build compiled for CUDA, found at run time without
//! naming the crate that compiled it.
//!
//! CUDA's compiler emits one [`CudaModelEntry`](crate::registry::CudaModelEntry) per model,
//! beside its `CUDA_TAPES` and `CUDA_WEIGHTS` (`scratchy-target-cuda-compiler`'s `emit`), and
//! submits it with `inventory::submit!`. The serving worker walks
//! [`entries`](crate::registry::entries) and keeps the first entry whose weight table a checkpoint
//! passes. Plain data: no toolchain, no GPU.

use crate::tape::CudaTape;
use crate::weights::CudaWeight;

/// One model as CUDA's compiler emitted it: its config stem, its tapes (ascending `bucket_m`) and
/// its weight table (per `WeightIx`).
#[derive(Debug)]
pub struct CudaModelEntry {
    pub stem: &'static str,
    pub tapes: &'static [CudaTape],
    pub weights: &'static [Option<CudaWeight>],
}

inventory::collect!(CudaModelEntry);

/// Every model registered in this binary, in no particular order.
pub fn entries() -> impl Iterator<Item = &'static CudaModelEntry> {
    inventory::iter::<CudaModelEntry>.into_iter()
}
