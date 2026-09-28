// SPDX-License-Identifier: Apache-2.0
//! METAL'S COMPILE-TIME LOWERING.
//!
//! ⭐ TARGET CODEGEN, IN THE TARGET'S DIRECTORY. `scratchy-forward-compiler-macro` depends on
//! `scratchy-target-metal`, so metal's expansion-time lowering could not live beside the runtime
//! and still name the compiler's handoff types without a cycle. This crate sits on the target's
//! side of that edge; the handoff types live in `scratchy_subtile::handoff`, which both sides see.
//! The macro calls ONE entry here per decode canonical, [`canonical::lower_canonical`], and the
//! bake, [`static_tape::bake_bucket_tapes`], per bucket.
//!
//! `syn`/`quote` are available here and NOT in `scratchy-subtile`, because this crate is
//! compile-time only and never links into the runtime.

/// One decode canonical, lowered: the entry the macro calls.
pub mod canonical;
/// Generic `const`-constructor emission for the metal tape types.
pub mod const_tokens;
/// The `const`-emitting bake: step tape -> per-bucket static tapes.
pub mod static_tape;
/// Metal's step records, built from the shared tape.
pub mod steps_from_tape;
/// The shared tape metal lowers from, re-rolled.
pub mod tape_program;
