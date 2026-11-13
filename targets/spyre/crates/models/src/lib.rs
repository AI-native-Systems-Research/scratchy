// SPDX-License-Identifier: Apache-2.0
// Generated code: the emitted modules mirror model stems and fused weight names, and bring the
// loader traits into scope whether or not a given model reads them.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! The selected models, compiled for spyre: one `pub mod <arch>` per arch the scope selects
//! (`scratchy-models`' features), each holding its models' baked bundles, launch wiring and
//! weights. Runtime discovery is via the emitted `inventory::submit!`
//! (`scratchy_target_spyre::SpyreArchRegistration`), so a consumer
//! `extern crate scratchy_target_spyre_models as _;` to force-link it.

// The tokenizers of the models this build compiles, baked by the build script from the Hub-cache
// checkpoints of those models (`scratchy_core_model::baked_tokenizer`).
include!(concat!(env!("OUT_DIR"), "/baked_tokenizers.rs"));

include!(concat!(env!("OUT_DIR"), "/modules.rs"));
