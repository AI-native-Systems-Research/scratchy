// SPDX-License-Identifier: Apache-2.0
//! Empty by design.
//!
//! This crate existed for its `build.rs`, which compiled the SuperDSC bundles the `#[forward]` macro
//! had dropped into a persistent `superdscforge` tree and generated a `baked_superdsc_dir(fp)` table
//! the runtime used to find them on disk.
//!
//! Both are gone. The emit compiles each launch group as it writes it
//! (`scratchy_target_spyre::superdsc_bake`) and the `#[forward]` macro bakes the result into the binary as
//! `scratchy_spyre_bundle::BundleCode` values, which the runtime looks up by fingerprint. Nothing is
//! cached, so there is no table to `include!` and no directory to look up.
//!
//! Kept as a build-graph placeholder so the `sendnn` / `superdsc` feature wiring that names this
//! crate keeps resolving; it goes when that wiring is untangled.
