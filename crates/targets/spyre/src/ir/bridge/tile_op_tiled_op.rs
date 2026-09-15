//! Bridge 2: `TileOp -> TiledOp` — the tiler. NOT pure syntax, and that is correct: core-split
//! (`WorkPlan::divide`) and LX-fit time-tiling (`WorkPlan::time_tile_for_lx`) are real scheduling
//! decisions. What makes this a disciplined bridge rather than a smuggled-in transform is that the
//! decision is CONFINED here — parameterized by an explicit, swappable `splitter` passed in by the
//! caller, never hardcoded per op kind, and never reaching backward into Bridge 1
//! (`subtile_tape_tile_op`, which stays pure syntax) or forward into Bridge 3's decomposition.

use crate::ir::island::tile_op::TileOp;
use crate::ir::island::tiled_op::TiledOp;
use scratchy_subtile::superdsc_opspec::{ItDim, MaxCores, WorkPlan};

/// Core-split, LX-fit time-tiling and the reduction-split repair that reconciles them, as ONE
/// target-neutral pass ([`WorkPlan::divide_and_time_tile_for_lx`]); this bridge supplies only the
/// swappable `splitter` and the residency formula this op's kind selects.
pub fn run_tiler<const N: u32>(
    op: &TileOp,
    budget: MaxCores<N>,
    splitter: impl Fn(&[ItDim], u32) -> std::collections::BTreeMap<&'static str, u32>,
    tiled_dim: &'static str,
) -> Result<TiledOp, scratchy_subtile::superdsc_error::SuperDscError> {
    WorkPlan::divide_and_time_tile_for_lx(&op.dims, budget, tiled_dim, splitter, |p, o| {
        op.resident_bytes(p, o)
    })
    .map(|(plan, time_tile)| TiledOp { plan, time_tile })
}
