//! Bridge 2: `TileOp -> TiledOp` — the tiler. NOT pure syntax, and that is correct: core-split
//! (`WorkPlan::divide`) and LX-fit time-tiling (`WorkPlan::time_tile_for_lx`) are real scheduling
//! decisions. What makes this a disciplined bridge rather than a smuggled-in transform is that the
//! decision is CONFINED here — parameterized by an explicit, swappable `splitter` passed in by the
//! caller, never hardcoded per op kind, and never reaching backward into Bridge 1
//! (`subtile_tape_tile_op`, which stays pure syntax) or forward into Bridge 3's decomposition.

use crate::ir::island::tile_op::TileOp;
use crate::ir::island::tiled_op::TiledOp;
use scratchy_subtile::superdsc_opspec::{ItDim, MaxCores, WorkPlan};

/// Run the full tiler over one `TileOp` declaration: core-split (proof (b)) then LX-fit
/// time-tiling — the two-step pipeline every `assemble_*` site hand-sequenced before `TileOp`
/// existed, now unified in one place. `TileOp::tile()` is a thin wrapper calling this.
pub fn run_tiler<const N: u32>(
    op: &TileOp,
    budget: MaxCores<N>,
    splitter: impl FnOnce(&[ItDim], u32) -> std::collections::BTreeMap<&'static str, u32>,
    tiled_dim: &'static str,
) -> Result<TiledOp, scratchy_subtile::superdsc_error::SuperDscError> {
    let plan = WorkPlan::divide(&op.dims, budget, splitter)
        .map_err(scratchy_subtile::superdsc_error::SuperDscError)?;
    let time_tile = plan.time_tile_for_lx(|p, opt| op.resident_bytes(p, opt), tiled_dim)?;
    Ok(TiledOp { plan, time_tile })
}
