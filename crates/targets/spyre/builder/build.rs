// SPDX-License-Identifier: Apache-2.0
//! Nothing to do at build time any more.
//!
//! This build script used to be the SuperDSC dxp bake: walk a persistent `superdscforge/<fp>/` tree
//! the `#[forward]` macro had filled, run `dxp_standalone` over every bundle in it, and generate a
//! `<fp> -> absolute directory` table the runtime read back off disk.
//!
//! The `#[forward]` emit now compiles each launch group itself, the moment it is written
//! (`scratchy_target_spyre::superdsc_bake`), and submits the compiled bytes to
//! the binary as `scratchy_spyre_bundle::BundleCode` values — so the device code is `&'static` data and
//! the staged compiler input is reclaimed as it is consumed. There is no directory to walk, no
//! fingerprint table to generate, and nothing left for a later build to find, prune or stale out.
//!
//! That also removes the reason this crate build-depended on `scratchy-models`. The edge existed ONLY
//! to guarantee the emit ran before this script (build-dependencies are the only ones cargo fully
//! builds first) — and it is what compiled every model TWICE, in two cargo units, doubling a ~13 GB
//! emission.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
