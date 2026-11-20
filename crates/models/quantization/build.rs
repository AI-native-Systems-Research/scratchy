// SPDX-License-Identifier: Apache-2.0
//! Publishes the enabled quant presets as `links` metadata, `DEP_SCRATCHY_QUANTIZATIONS_PRESETS`
//! in the build script of every crate that depends on this one directly. That is how
//! `scratchy-models`' build script learns the quant scope: a `quant/<preset>` feature a CLI turns
//! on through an ordinary dependency lands on this same build of the crate.

include!("src/presets.rs");

/// The names of the presets whose same-named Cargo feature is active on this build, in name order:
/// the one place the scope leaves the crate as text, because build metadata is text.
macro_rules! enabled_names {
    ($($variant:ident $name:literal $scheme:ident,)*) => {
        fn enabled_names() -> Vec<&'static str> {
            let mut names = Vec::new();
            $(if cfg!(feature = $name) {
                names.push($name);
            })*
            names
        }
    };
}

with_presets!(enabled_names);

fn main() {
    println!("cargo::metadata=presets={}", enabled_names().join(","));
}
