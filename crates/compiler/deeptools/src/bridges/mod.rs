//! ONE MODULE PER BRIDGE, named `<from>_to_<to>` — where two vocabularies meet.
//!
//! ⭐ THE DDL TEMPLATE IS THE SCHEDULE (units, transfers, loop nest, computes) — the same for
//! every op of its op-func, and it knows no extents. THE NODE IS THE SHAPE. Two sides, both
//! needed; a bridge that derives one from the other has invented it.

/// The subtile tape becoming DataflowIR.
pub mod subtile_to_dataflow_ir;

