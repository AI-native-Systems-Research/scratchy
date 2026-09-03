// SPDX-License-Identifier: Apache-2.0
//! METAL'S COMPILE-TIME LOWERING.
//!
//! ⭐ TARGET CODEGEN, IN THE TARGET'S DIRECTORY. Both modules here were in
//! `crates/compiler/macros/src/` (`metal_from_subtile.rs`, `metal_static_tape.rs`) — metal's own
//! instruction selection and static-tape bake, living in the shared compiler crate. They could
//! not simply move beside the rest of metal because `scratchy-forward-compiler-macro` depends on
//! `scratchy-target-metal`, so the target side could not name the compiler's handoff types
//! without a cycle. This crate sits on the target's side of that edge; the handoff types moved
//! down to `scratchy_subtile::handoff`, which both sides can see.
//!
//! `syn`/`quote` are available here and NOT in `scratchy-subtile`, because this crate is
//! compile-time only and never links into the runtime.

/// Generic `const`-constructor emission for the metal tape types.
pub mod const_tokens;
/// The bridge from the shared `SubtileTape` to metal's instruction stream.
pub mod from_subtile;
/// The `const`-emitting bake: instruction stream -> per-bucket static tapes.
pub mod static_tape;
/// The shared tape metal lowers from, re-rolled.
pub mod tape_program;
