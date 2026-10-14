// SPDX-License-Identifier: Apache-2.0
//! Integration-test helpers: a re-export of the crate's MTL4 single-op dispatch helper
//! ([`scratchy_target_metal::mtl4_dispatch`]), reached as `common::dispatch_threadgroups`,
//! `common::shared_slice`, `common::Buffer`, etc., and the kernel tests' baked pipeline builds
//! ([`baked_build`], [`baked_pipeline`]). Declared `pub mod common` so a test binary that uses a
//! subset of the helpers leaves the rest unused without a dead-code warning.

use std::sync::Mutex;

use scratchy_target_metal::aot::{baked_kernels, baked_library};
use scratchy_target_metal::specialized_pipeline_cache::{
    ComputePipelineState, ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::stream::MetalStreamError;

pub use scratchy_target_metal::mtl4_dispatch::*;

/// `key`'s pipeline from `cache`, its kernel baked first — for kernel tests that build several
/// pipelines through one cache.
///
/// One build at a time, process-wide: parallel tests each build through their own cache (and
/// `MTL4Compiler`), and AGX's compiler corrupts the heap when those builds overlap (#288). The
/// serving pool builds on one thread, so production's `get_or_build` takes no lock.
pub fn baked_build(
    cache: &SpecializedPipelineCache,
    key: &PipelineKey,
) -> Result<ComputePipelineState, MetalStreamError> {
    if baked_library(key.library_name).is_some() {
        cache.register_baked(&baked_kernels(std::slice::from_ref(key)));
    }
    static BUILD: Mutex<()> = Mutex::new(());
    let _build = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    cache.get_or_build(key)
}

/// A kernel test's baked pipeline, held with the cache that built it: a pipeline used after its
/// cache (MTL4 compiler, library) is dropped computes garbage.
pub struct BakedPipeline {
    pipeline: ComputePipelineState,
    _cache: SpecializedPipelineCache,
}

impl std::ops::Deref for BakedPipeline {
    type Target = ComputePipelineState;
    fn deref(&self) -> &ComputePipelineState {
        &self.pipeline
    }
}

/// The pipeline of `library`'s kernel `function` with `constants`, baked and built the way the
/// pool builds it — for kernel tests.
pub fn baked_pipeline(
    device: &scratchy_target_metal::specialized_pipeline_cache::Device,
    library: &'static str,
    function: &str,
    constants: Vec<ConstantValue>,
) -> Result<BakedPipeline, MetalStreamError> {
    let key = PipelineKey::new(library, Box::leak(Box::from(function)), constants);
    let cache = SpecializedPipelineCache::new(device.clone(), &[])?;
    let pipeline = baked_build(&cache, &key)?;
    Ok(BakedPipeline {
        pipeline,
        _cache: cache,
    })
}
