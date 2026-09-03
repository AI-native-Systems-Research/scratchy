//! The SuperDSC compilation pipeline, made LITERAL in module structure: every IR level is an
//! `island::*` module (a stable data type), every translation between two levels is a `bridge::x_y`
//! module (a function from one island's type to the next's). The naming is the invariant:
//!
//! ```text
//!   island::subtile_tape   (SubtileTape  — pre-existing, shared with the cuda megakernel pipeline)
//!         │
//!         ▼  bridge::subtile_tape_tile_op   (node_to_tile_ops — PURE SYNTAX, see its doc)
//!   island::tile_op         (TileOp — "what to compute, at what shape")
//!         │
//!         ▼  bridge::tile_op_tiled_op        (the tiler — core-split + LX-fit time-tiling)
//!   island::tiled_op        (TiledOp — a concrete schedule: WorkPlan + optional TimeTile)
//!         │
//!         ▼  bridge::tiled_op_sdsc_op        (decomposition INTO concrete device ops)
//!   island::sdsc_op         (OpSpec/EmittedOp — pre-existing, the wire format)
//! ```
//!
//! Two rules this structure exists to enforce (both learned the hard way this session):
//!
//! 1. **A `bridge::x_y` module either translates or transforms — never both silently.**
//!    `bridge::subtile_tape_tile_op` is PURE SYNTAX: it reads a node's own `op` kind, `output.region`
//!    shape, and `inputs.len()`, and produces ONE `TileOp` — no branch on row count, no batching
//!    decision, no decomposition. `bridge::tile_op_tiled_op` (the tiler) is NOT pure syntax and that is
//!    correct: scheduling (core-split, time-tiling) is a real decision, but it is CONFINED to this one
//!    bridge, parameterized by an explicit, swappable `splitter` — never smuggled into the syntax
//!    bridge on either side of it.
//!
//! 2. **Decomposition of a composite op (RmsNorm's 6 steps, RoPE's rotate+xc+rs+add, AttnDecode's
//!    fused attention) belongs on `bridge::tiled_op_sdsc_op`, not earlier.** A fused op's tiler
//!    decision (does it fit LX? how many cores?) depends on its AGGREGATE resident footprint, which
//!    the decomposition eventually produces — so the schedule must be decided FIRST (against the whole
//!    composite `TileOp`), and decomposition into concrete ops happens AFTER, informed by that
//!    schedule. Decomposing before tiling would make each piece guess at a schedule the whole
//!    computation hasn't settled on yet.
//!
//! `island::subtile_tape` and `island::sdsc_op` are THIN RE-EXPORT SHIMS pointing at their real,
//! pre-existing definitions (`scratchy_subtile::subtile_tape`, `scratchy_subtile::superdsc_opspec`/
//! `crate::lower_subtile_tape_to_superdsc::EmittedOp`) — those types are NOT owned by this tree and are
//! shared with code outside it (the cuda megakernel pipeline reads `SubtileTape` directly). Only
//! `island::tile_op` and `island::tiled_op` are new types that live here for real.

pub mod bridge;
pub mod island;
