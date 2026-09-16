//! Every stable data type in the SDSC pipeline, one per pipeline stage. See `crate::ir` for the
//! full map. `subtile_tape` and `sdsc_op` are re-export shims (those types are owned elsewhere and
//! shared outside this tree); `tile_op` and `tiled_op` are real, new homes.

pub mod sdsc_op;
pub mod subtile_tape;
// ⭐ `tile_op` AND `tiled_op` NOW LIVE IN `ktir-superdsc` — see that crate's `ir` module for why the
// same `ir/{island,bridge}` shape was rebuilt there. Re-exported as MODULES, so every
// `crate::ir::island::tile_op::TileOp` path in this crate resolves unchanged and the two files moved
// with `git mv` (100% rename) rather than being retyped.
pub use ktir_superdsc::ir::island::{tile_op, tiled_op};
