// SPDX-License-Identifier: Apache-2.0
//! The typed program: every carrier the typed-Rust emit pass (the front end's `typed_emit`, run by
//! the build script) turned into a Rust function over the op signatures (`scratchy-typed-shapes`).
//! A shape error in a carrier is a type error at the generated line, which names the carrier's file
//! and line; the functions also RUN, and [`traced`] is the shapes they record.

use scratchy_typed_shapes::*;

// The decoder states the build compiles (`decoder_states()`): the build's list, not the typed program.
mod decoder_states;
pub use decoder_states::decoder_states;
// The typing every build hands the forward compiler (`typing()`): the typed programs under that list.
mod typing;
pub use typing::typing;

// The emit pass's output: the marker of every bound the carriers' weights name that the signatures
// do not (`typed_emit::SIGNATURE_BOUNDS`), one module per carrier, `typed_program`, which names each
// carrier's traced function by its arch, and `derived_dims_of`.
include!(concat!(env!("OUT_DIR"), "/typed_carriers.rs"));

/// The shapes of the carrier whose arch is `arch`: its typed program, run. `None` when the emit
/// pass typed no carrier of that arch.
pub fn traced(arch: &str) -> Option<Shapes> {
    Some(scratchy_typed_shapes::trace(typed_program(arch)?))
}
