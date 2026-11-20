// SPDX-License-Identifier: Apache-2.0
//! THE CUDA BACKEND'S SERVING WORKER — what `cuda chat` and `cuda serve` build.
//!
//! The serving engine builds its worker through the neutral `WorkerFactory` registry: a binary
//! links exactly one target's worker, and the engine calls the one registered factory's `create`
//! (`linked_factory`). [`CudaWorkerFactory`] is CUDA's entry, registered below with
//! `inventory::submit!`; the `cuda` binary anchors this crate, so no shared code names it.
//!
//! The worker it builds (`serving`, under the `nvcc` feature) runs one sequence at a time with
//! greedy sampling. It finds its model in the CUDA model registry
//! (`scratchy_target_cuda::registry`, filled by the emitted per-model entries), loads its weights
//! with the shared `runtime::load_weights`, and per scheduler step plays one of the model's tapes
//! with `CudaWorker::forward`. What a step plays is planned on the host by [`plan`], which needs
//! no toolchain.
//!
//! Without `nvcc` the factory still registers, and its `create` refuses with the rebuild command,
//! before it resolves any model path.

// Force the linker to keep the compiled models (and thus their `CudaModelEntry`
// `inventory::submit!`s): otherwise, referenced only through the registry, they get dead-stripped
// and the worker finds an empty registry.
extern crate scratchy_target_cuda_models as _;

pub mod plan;
#[cfg(feature = "nvcc")]
mod serving;

use scratchy_core_common::KvBlockTokens;
use scratchy_serving_engine::worker_factory::{
    ProgressCallback, Served, WorkerCreateConfig, WorkerCreationResult, WorkerFactory,
};
use scratchy_target_cuda::step::CUDA_KV_BLOCK_SIZE;

/// What a build without the `nvcc` feature says when asked for a CUDA worker.
pub const NO_NVCC: &str = "this cuda was built without the `nvcc` feature, so the CUDA kernels are \
     not linked; rebuild on a Linux host with CUDA 12.8+, in `backends/cuda`: `cargo build --release \
     --features nvcc,model/<stem>,quant/fp8-dynamic-per-channel`";

/// The CUDA backend's factory.
pub struct CudaWorkerFactory;

impl WorkerFactory for CudaWorkerFactory {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn create(
        &self,
        cfg: WorkerCreateConfig,
        _progress: Option<ProgressCallback>,
    ) -> anyhow::Result<WorkerCreationResult> {
        create(cfg)
    }

    /// No: the worker keeps no snapshot of the model's Gated-DeltaNet state, and a recurrent
    /// hybrid cannot resume from a cached KV prefix without one.
    fn supports_prefix_caching(&self) -> bool {
        false
    }

    /// The KV block is the attention kernel's 64-token tile (`CUDA_KV_BLOCK_SIZE`), which the
    /// scheduler's `slot = block_id * block_size + offset` must address too.
    fn required_block_size(&self) -> Option<KvBlockTokens> {
        KvBlockTokens::new(CUDA_KV_BLOCK_SIZE as usize)
    }

    /// Greedy decoding of one completion, with no penalty, logprobs or logit transform: what the
    /// worker's one tape plays (a greedy argmax of the raw logits, one sequence at a time). The
    /// API refuses anything else at its door ([`Served::refusal`]), so the scheduler never admits
    /// it.
    fn served(&self) -> Served {
        Served {
            sampling: false,
            penalties: false,
            completions: false,
            logprobs: false,
            logit_transforms: false,
        }
    }
}

/// Without the kernels there is no worker to build.
#[cfg(not(feature = "nvcc"))]
fn create(_cfg: WorkerCreateConfig) -> anyhow::Result<WorkerCreationResult> {
    anyhow::bail!("{NO_NVCC}")
}

#[cfg(feature = "nvcc")]
fn create(cfg: WorkerCreateConfig) -> anyhow::Result<WorkerCreationResult> {
    serving::create(cfg)
}

inventory::submit!(&CudaWorkerFactory as &dyn WorkerFactory);
