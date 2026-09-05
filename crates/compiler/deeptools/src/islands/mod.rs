//! THE ISLANDS — one per IR the backend compiler's ladder passes through.
//!
//! ```text
//! SubtileIR tape ──► DataflowIR ──► SentientIR ──► ProgIR ──► SenProg ──► init_binary
//!    (scratchy)        (here)
//! ```
//!
//! ⭐ AN ISLAND IS A REPRESENTATION AND NOTHING ELSE: its types, its invariants, and how it is
//! written out. It knows nothing about what produced it or what consumes it — that is the
//! [`crate::bridges`]' job, and keeping the two apart is what stops a lowering decision from being
//! made inside a data type where no test would find it.
//!
//! Only DataflowIR is built. `SenProg` is already ported elsewhere in the tree; the three between
//! are the remaining rungs.

pub mod dataflow_ir;
