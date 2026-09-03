// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral neural-network layer types.
//!
//! Holds the layer *type definitions* and their *pure* accessors (the surface
//! the metal interpreter and `WeightAccessors` match on) so both
//! `targets/{cuda,metal}` can name them without a circular dependency. CUDA/Metal
//! runtime methods (kernel-calling `forward`, `GpuWeights`-bound `load`) stay in
//! the target crates as extension traits — this crate names no backend.

pub mod gdn_state;
pub mod kv_cache;
pub mod layers;
pub mod layers_moe;
pub mod loaders;
pub mod rotary;
pub mod turboquant;
pub mod weights;

// Neutral tensor-core *modules* re-exported so this crate can serve as the
// backend-less `__gpu` alias target the forward-compiler macro emits against for
// arches compiled without a backend feature (e.g. phi3 / deepseek under
// `--features metal`). Those builds reference the tensor core only by module
// path (`__gpu::tensor::…` / `__gpu::dtype::DType`), so mirroring the modules
// (not the bare type names, which would collide with the targets' own glob
// re-exports) is sufficient.
pub use scratchy_tensors::{device_allocator, dtype, tensor};

pub use gdn_state::{GDN_STATE_DTYPE, GdnStatePool};
pub use kv_cache::KvCachePool;
pub use layers::{
    Bnb4bitLinear, CohereLayerNorm, Embedding, Fp8AnyLinear, Fp8BlockLinear, Fp8Linear,
    GatedDeltaNetLayer, GgmlLinear, LayerNorm, LayerNormBias, Linear, LinearLayer, MarlinLinear,
    RmsNorm,
};
pub use weights::{GpuWeights, UploadSrc};
