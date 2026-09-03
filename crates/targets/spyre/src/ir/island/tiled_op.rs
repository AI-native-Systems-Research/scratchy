//! `TiledOp` — the tiler's output (Bridge 2: `crate::ir::bridge::tile_op_tiled_op`) for ONE
//! `TileOp`. The pipeline's third named stage: `SubtileTape → TileOp (TileIR) → tiler → TiledOp →
//! SdscOp`. Every TileIR→SdscOp lowering (`pointwise_opspec_from_tile` and its siblings in
//! `lower_subtile_tape_to_superdsc.rs`) takes a `TiledOp`, not a raw `(WorkPlan, Option<TimeTile>)`
//! tuple, so the stage boundary is a real type, not a convention.

use scratchy_subtile::superdsc_opspec::{TimeTile, WorkPlan};

pub struct TiledOp {
    pub plan: WorkPlan,
    pub time_tile: Option<TimeTile>,
}
