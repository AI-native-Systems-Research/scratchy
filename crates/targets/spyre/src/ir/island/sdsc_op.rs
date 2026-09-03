//! Re-export shim: the wire-format op types (`OpSpec`/`EmittedOp` and friends) are owned by
//! `scratchy_subtile::superdsc_opspec` and `crate::lower_subtile_tape_to_superdsc`, not this tree — they
//! predate TileIR and are the device-facing format the whole emitter targets. This module exists
//! only so the pipeline map in `crate::ir` names the final stage as a real island.

pub use scratchy_subtile::superdsc_opspec::*;
