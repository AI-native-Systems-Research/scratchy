// SPDX-License-Identifier: Apache-2.0
//! CUDA kernel compilation for the scratchy inference framework.
//!
//! This crate has no runtime code. It exists solely for its `build.rs`,
//! which compiles all static kernel `.cu` files into static `.a`
//! libraries. The dependency on `scratchy-models` ensures that
//! `forward!()` macro expansion runs first.

// Re-export scratchy-models so downstream crates get the build ordering.
pub use scratchy_models;

// Per-tuple FlashInfer config set — symbol-name source of truth shared
// with `build.rs` (via `#[path]`) and consumed downstream by
// scratchy-target-cuda / scratchy-forward-compiler-macro when emitting FI call sites.
pub mod flashinfer_config;
