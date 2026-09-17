// SPDX-License-Identifier: Apache-2.0
//! Generated chat-template renderers, one module per vendored template.
//!
//! Nothing is hand-written here. `build.rs` compiles each vendored `.jinja` into
//! Rust and this file `include!`s the result — the same shape the models crate
//! will use.
include!(concat!(env!("OUT_DIR"), "/templates.rs"));
