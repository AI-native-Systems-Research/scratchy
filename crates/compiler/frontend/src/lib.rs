// SPDX-License-Identifier: Apache-2.0
//! The forward compiler's front end: a carrier's Python costume parsed ([`parse_python`], into
//! [`ast`]), classified into the program the rest of the compiler reads ([`classify`],
//! [`classified`]), the drafter capture every target gets ([`draft_capture`]), the dims its shapes
//! are made of ([`dim`]), the config rules ([`hf_config`]), each arch's weights manifest
//! ([`weights_manifest`]), and the typed-Rust emit pass over all of it ([`typed_emit`]).
//!
//! One crate so it compiles once: `scratchy-forward-compiler-macro` runs the pipeline on it, and
//! that crate's build script runs the emit pass on it before the macro crate compiles.

pub mod ast;
pub mod bundles;
pub mod carrier_facts;
pub mod classified;
pub mod classify;
pub mod dim;
pub mod draft_capture;
pub mod hf_config;
pub mod parse_python;
pub mod typed_emit;
pub mod weights_manifest;
