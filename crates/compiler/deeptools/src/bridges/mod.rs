//! ONE MODULE PER BRIDGE, named `<from>_to_<to>` — where two vocabularies meet.
//!
//! ⭐ THE DDL TEMPLATE IS THE SCHEDULE (units, transfers, loop nest, computes) — the same for
//! every op of its op-func, and it knows no extents. THE NODE IS THE SHAPE. Two sides, both
//! needed; a bridge that derives one from the other has invented it.

/// The subtile tape becoming DataflowIR.
pub mod subtile_to_dataflow_ir;

/// `DataflowIR -> SentientIR` — the D1-D28 span.
pub mod dataflow_ir_to_sentient;

/// `SentientIR -> ProgIR` — dcc pass D76, the last MLIR rung.
pub mod sentient_to_progir;
