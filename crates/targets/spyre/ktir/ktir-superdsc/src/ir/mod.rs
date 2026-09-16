// SPDX-License-Identifier: Apache-2.0
//! THE PIPELINE, MADE LITERAL IN MODULE STRUCTURE — every IR level is an `island::*` module (a
//! stable data type), every translation between two levels is a `bridge::x_y` module (a function
//! from one island's type to the next's).
//!
//! ⭐ THE PATH TEXT IS THE REASON THIS SHAPE CAME ACROSS UNCHANGED. These modules were
//! `scratchy-target-spyre`'s `crate::ir::island::tile_op` / `crate::ir::bridge::tile_op_tiled_op`,
//! and they name each other by those exact paths. Rebuilding the same `ir/{island,bridge}` tree here
//! means every one of those `crate::ir::…` references still resolves, spelled the same way — so the
//! move cost zero path edits inside the moved files, and `scratchy-target-spyre` re-exports the four
//! modules so its own `crate::ir::island::tile_op::TileOp` paths keep resolving too.
//!
//! ⛔ THE ASSEMBLERS' INPUT TYPE WAS IN NO CLASS THE EXTRACTION PLAN DREW, AND THE AUDIT CORRECTED
//! IT. The plan said the lowering's input is a `KtirNode`, which is true of the *lowering*; but
//! every pointwise/reduce/matmul assembler under it takes a `&TileOp`, so `TileOp`/`TileOpKind`/
//! `TiledOp` and the tiler between them had to come too.

/// The two translations that travel with the island: the tiler (core-split + LX-fit time-tiling)
/// and the span-overflow refusal it consults.
pub mod bridge;
/// The stable data types: [`island::tile_op::TileOp`] ("what to compute, at what shape") and
/// [`island::tiled_op::TiledOp`] (a concrete schedule: `WorkPlan` + optional `TimeTile`).
pub mod island;
