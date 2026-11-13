// SPDX-License-Identifier: Apache-2.0
//! METAL'S COMPILE-TIME LOWERING AND EMISSION.
//!
//! ⭐ TARGET CODEGEN, IN THE TARGET'S DIRECTORY. The shared pipeline
//! (`scratchy-forward-compiler-macro`) calls a target through its `TargetEmitter`; [`Metal`] is
//! metal's. The pipeline hands it each canonical model ([`model`]: buckets, baked tapes, arenas,
//! the pool and the forward entry points) and each arch ([`dispatch`]: the loaded-model surface
//! and registry rows); the lowering under them lives here too — [`canonical::lower_canonical`]
//! per decode canonical, and [`static_tape::bake_bucket_tapes`] per bucket. The handoff types the
//! lowering shares with the runtime live in `scratchy_subtile::handoff`.
//!
//! `syn`/`quote` are available here and NOT in `scratchy-subtile`, because this crate is
//! compile-time only and never links into the runtime.

/// One decode canonical, lowered: the entry the per-model emission calls.
pub mod canonical;
/// Generic `const`-constructor emission for the metal tape types.
pub mod const_tokens;
/// An arch's loaded-model surface on metal.
mod dispatch;
/// The emitter the shared pipeline calls.
mod emitter;
/// One canonical model's emission on metal.
mod model;
/// The `const`-emitting bake: step tape -> per-bucket static tapes.
pub mod static_tape;
/// Metal's step records, built from the shared tape.
pub mod steps_from_tape;
/// The shared tape metal lowers from, re-rolled.
pub mod tape_program;

pub use emitter::Metal;
