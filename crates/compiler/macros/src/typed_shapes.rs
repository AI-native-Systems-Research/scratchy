// SPDX-License-Identifier: Apache-2.0
//! The typed program: the signatures (`scratchy-typed-shapes`, re-exported here) with every carrier
//! the typed-Rust emit pass (`typed_emit.rs`, run by the build script) turned into a Rust function
//! over them. A shape error in a carrier is a type error at the generated line, which names the
//! carrier's file and line; the functions also RUN, and [`traced`] is the shapes they record.

pub use scratchy_typed_shapes::*;

// The emit pass's output: the marker of every bound the carriers' weights name that the signatures
// do not (`typed_emit::SIGNATURE_BOUNDS`), one module per carrier, and
// `typed_program`, which names each carrier's traced function by its arch.
include!(concat!(env!("OUT_DIR"), "/typed_carriers.rs"));

/// The shapes of the carrier whose arch is `arch`: its typed program, run. `None` when the emit
/// pass typed no carrier of that arch.
pub fn traced(arch: &str) -> Option<Shapes> {
    Some(scratchy_typed_shapes::trace(typed_program(arch)?))
}
