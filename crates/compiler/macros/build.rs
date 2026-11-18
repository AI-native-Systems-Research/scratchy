// SPDX-License-Identifier: Apache-2.0
//! The compiler this crate builds with, for its test support: `src/typed_bodies.rs` types a
//! program of its own with it, the way `scratchy-models-typed` types the carriers.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rustc-env=TYPED_SHAPES_RUSTC={}",
        std::env::var("RUSTC").expect("RUSTC")
    );
}
