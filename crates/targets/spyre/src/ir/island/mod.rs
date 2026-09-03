//! Every stable data type in the SDSC pipeline, one per pipeline stage. See `crate::ir` for the
//! full map. `subtile_tape` and `sdsc_op` are re-export shims (those types are owned elsewhere and
//! shared outside this tree); `tile_op` and `tiled_op` are real, new homes.

pub mod sdsc_op;
pub mod subtile_tape;
pub mod tile_op;
pub mod tiled_op;
