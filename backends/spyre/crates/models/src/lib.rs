// SPDX-License-Identifier: Apache-2.0
//! The selected models, compiled for spyre: one `pub mod <arch>` per arch the scope selects
//! (`scratchy-models`' features), each holding its models' baked bundles, launch wiring and
//! weights. Runtime discovery is via the emitted `inventory::submit!`
//! (`scratchy_target_spyre::SpyreArchRegistration`), so a consumer
//! `extern crate scratchy_target_spyre_models as _;` to force-link it.

// The tokenizers of the models this build compiles, baked by the build script from the Hub-cache
// checkpoints of those models (`scratchy_core_model::baked_tokenizer`).
include!(concat!(env!("OUT_DIR"), "/baked_tokenizers.rs"));

include!(concat!(env!("OUT_DIR"), "/modules.rs"));
