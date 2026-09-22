// SPDX-License-Identifier: Apache-2.0
//! CUDA kernel compilation for the scratchy inference framework.
//!
//! This crate has no runtime code. It exists solely for its `build.rs`,
//! which compiles all static kernel `.cu` files into static `.a`
//! libraries.
//!
//! It deliberately does NOT depend on `scratchy-models`: that crate depends on
//! `scratchy-target-cuda`, whose rlib bundles the `.a` files this `build.rs`
//! produces, so the edge was a bootstrap cycle. Build this crate FIRST — see
//! `docs/BUILD.md`.

// Per-tuple FlashInfer config set — symbol-name source of truth shared
// with `build.rs` (via `#[path]`) and consumed downstream by
// scratchy-target-cuda / scratchy-forward-compiler-macro when emitting FI call sites.
pub mod flashinfer_config;
