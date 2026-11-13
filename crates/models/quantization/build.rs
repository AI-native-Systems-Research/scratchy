// SPDX-License-Identifier: Apache-2.0
//! Publishes the enabled quant presets as `links` metadata, `DEP_SCRATCHY_QUANTIZATIONS_PRESETS`
//! in the build script of every crate that depends on this one directly. That is how
//! `scratchy-models`' build script learns the quant scope: a `quant/<preset>` feature a CLI turns
//! on through an ordinary dependency lands on this same build of the crate.

include!("src/presets.rs");

fn main() {
    println!("cargo::metadata=presets={}", enabled_presets().join(","));
}
