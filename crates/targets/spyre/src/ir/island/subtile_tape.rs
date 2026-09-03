//! Re-export shim: `SubtileTape` is owned by `scratchy_subtile::subtile_tape`, not this tree — it is shared
//! with the cuda megakernel pipeline, which reads it directly and has no notion of TileIR. This
//! module exists only so the pipeline map in `crate::ir` names it as a real stage.

pub use scratchy_subtile::subtile_tape::*;
