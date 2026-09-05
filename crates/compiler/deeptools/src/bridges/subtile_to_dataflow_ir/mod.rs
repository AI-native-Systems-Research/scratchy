//! BRIDGE 1 — the subtile tape's vocabulary meeting DataflowIR's.
//!
//! ⭐ THE DDL TEMPLATE IS THE SCHEDULE, THE NODE IS THE SHAPE. The template says which units take
//! part, what moves between them and in what loop nest; it knows no extents. The node says how big
//! everything is. A bridge that derives one from the other has invented it.
//!
//! ⛔ WHAT IS HERE SO FAR IS THE TRANSFER PLANNER, and it takes no scratchy type: it is written in
//! extents and a vector width, so it can be checked against IBM's own DataflowIR without a tape.
//! The tape-facing half comes next.

/// One node of the tape, in a vocabulary this crate can name.
pub mod node;

/// The template's schedule, walked.
pub mod schedule;

/// The whole forward tape, compiled.
pub mod tape;

/// How a transfer is walked over the AGEN time axis.
pub mod transfer;
