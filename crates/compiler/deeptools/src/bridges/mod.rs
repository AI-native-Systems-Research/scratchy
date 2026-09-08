//! ONE MODULE PER BRIDGE, named `<from>_to_<to>` — where two vocabularies meet.
//!
//! ⭐ THE DDL TEMPLATE IS THE SCHEDULE (units, transfers, loop nest, computes) — the same for
//! every op of its op-func, and it knows no extents. THE NODE IS THE SHAPE. Two sides, both
//! needed; a bridge that derives one from the other has invented it.

/// The subtile tape becoming DataflowIR.
pub mod subtile_to_dataflow_ir;

/// `DataflowIR -> SentientIR` — the D1-D28 span.
pub mod dataflow_ir_to_sentient;

/// `SuperDSC -> DataflowIR` — the ported conversion that replaces `subtile_to_dataflow_ir`.
pub mod superdsc_to_dataflow_ir;
