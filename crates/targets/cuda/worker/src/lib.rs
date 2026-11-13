// SPDX-License-Identifier: Apache-2.0
//! THE CUDA BACKEND'S SERVING WORKER — what `scr chat --device cuda` and `scr serve --device
//! cuda` build.
//!
//! The serving engine builds its worker through the neutral `WorkerFactory` registry: it walks the
//! registered factories, takes the first whose `matches` accepts the `--device` string, and calls
//! its `create`. [`CudaWorkerFactory`] is CUDA's entry, registered below with `inventory::submit!`;
//! the serving worker crate links this crate under its neutral `backend-worker` key, so no shared
//! code names it.
//!
//! The worker it builds (`serving`, under the `nvcc` feature) runs one sequence at a time with
//! greedy sampling. It finds its model in the CUDA model registry
//! (`scratchy_target_cuda::registry`, filled by the emitted per-model entries), loads its weights
//! with the shared `runtime::load_weights`, and per scheduler step plays one of the model's tapes
//! with `CudaWorker::forward`. What a step plays is planned on the host by [`plan`], which needs
//! no toolchain.
//!
//! Without `nvcc` the factory still registers and claims `--device cuda` (and `auto`), and its
//! `create` refuses with the rebuild command, before it resolves any model path.

pub mod plan;
#[cfg(feature = "nvcc")]
mod serving;

use scratchy_core_common::{KvBlockTokens, SamplingParams};
use scratchy_serving_engine::worker_factory::{
    ProgressCallback, WorkerCreateConfig, WorkerCreationResult, WorkerFactory,
};
use scratchy_target_cuda::step::CUDA_KV_BLOCK_SIZE;

/// What a build without the `nvcc` feature says when asked for a CUDA worker.
pub const NO_NVCC: &str = "this scr was built without the `nvcc` feature, so the CUDA kernels are \
     not linked; rebuild on a Linux host with CUDA 12.8+: `cargo build --release -p scratchy-cli \
     --features nvcc,model/<stem>`";

/// The CUDA backend's factory: `--device cuda`, and `auto`.
pub struct CudaWorkerFactory;

impl WorkerFactory for CudaWorkerFactory {
    fn matches(&self, device: &str) -> bool {
        // ⭐ ONE DEVICE NAME. Only one backend compiles per build (the `#[forward]` macro refuses
        // two), so `auto` resolves to whichever backend this binary was built with, and in a CUDA
        // build that is this one.
        device == "cuda" || device == "auto"
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

    /// Greedy only ([`plan::check_sampling`]): the engine refuses anything else when the request
    /// is submitted, so the scheduler never admits it.
    fn check_sampling(&self, params: &SamplingParams) -> Result<(), String> {
        plan::check_sampling(Some(params)).map_err(|e| e.to_string())
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
