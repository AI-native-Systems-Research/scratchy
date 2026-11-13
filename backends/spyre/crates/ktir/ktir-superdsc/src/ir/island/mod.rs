// SPDX-License-Identifier: Apache-2.0
//! The stable data types of the TileIR levels. See [`super`] for the pipeline map.
//!
//! `subtile_tape` and `sdsc_op` — the two shim modules this tree had in `scratchy-target-spyre` —
//! did NOT come with these two. They point at `scratchy_subtile::subtile_tape` and at the emitter's
//! `EmittedOp`, i.e. at the tape side and the carrier side, neither of which is this crate's.

/// "What to compute, at what shape" — one op, before any scheduling decision.
pub mod tile_op;
/// A concrete schedule: a `WorkPlan` and an optional `TimeTile`.
pub mod tiled_op;
