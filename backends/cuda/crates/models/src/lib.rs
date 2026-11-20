// SPDX-License-Identifier: Apache-2.0
// Generated code: the emitted modules mirror model stems, and bring the tape modules into scope
// whether or not a given tape reads them.
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
//! The selected models, compiled for CUDA: one `pub mod <arch>` per arch the scope selects
//! (`scratchy-models`' features) that CUDA lowers, each holding its models' `static` tapes
//! (`CUDA_TAPES`), weight tables (`CUDA_WEIGHTS`) and registry entries. The worker finds them
//! through the emitted `inventory::submit!` (`scratchy_target_cuda::registry::CudaModelEntry`), so
//! a consumer `extern crate scratchy_target_cuda_models as _;` to force-link it.

// The model scope: `scratchy-models` is read only through its build metadata, which this crate's
// build script reads, so nothing names it and cargo::unused_dependencies flags it unless this does.
extern crate scratchy_models as _;

// The tokenizers of the models this build compiles, baked by the build script from the Hub-cache
// checkpoints of those models (`scratchy_core_model::baked_tokenizer`).
include!(concat!(env!("OUT_DIR"), "/baked_tokenizers.rs"));

include!(concat!(env!("OUT_DIR"), "/modules.rs"));
