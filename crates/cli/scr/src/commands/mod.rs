// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! CLI subcommand implementations.

#[cfg(any(feature = "chat", feature = "serve"))]
pub mod batch;
#[cfg(feature = "bench")]
pub mod bench;
// ⛔ GATED, because every item `bundle.rs` names is. It imports `BundleRunArgs` and
// `scratchy_target_spyre`, both `#[cfg(feature = "spyre-hw")]`, so an ungated declaration breaks the
// DEFAULT build and `-Fcuda`/`-Fmetal` and CI with `E0432`/`E0433` — measured `rc 101` on
// `cargo check -p scratchy-cli`, against `rc 0` on main.
#[cfg(feature = "spyre-hw")]
pub mod bundle;
pub mod cache;
pub mod chat;
pub mod collect_env;
pub mod completions;
pub mod convert;
#[cfg(feature = "claude")]
pub mod launch;
pub mod model;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod model_info;
pub mod pull;
#[cfg(feature = "serve")]
pub mod serve;
