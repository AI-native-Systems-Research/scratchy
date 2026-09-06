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
//! ⭐⭐ THE LADDER'S RUNGS, MEASURED RATHER THAN NAMED. `dbo/docs/pass_pipeline.md` on the pod is the
//! authority: `buildDSCToSentientIRPipeline` (D1-D28) reaches [`sentient`], D29-D75 rewrite it in
//! place, and D76 `SentientToProgIR` reaches [`progir`].
//!
//! ⛔ AND `SenProg` IS NOT A RUNG. It is one of three *print formats* of ProgIR, selected inside
//! `SentientToProgIR.cpp:703-737`'s `if (dumpProgIR.getValue())`; what dip consumes is the ProgIR
//! structure itself (`GenerateInitPacket.cpp:48`). So the ladder is three islands, not four, and
//! senprog is an oracle to diff against — see [`progir::print`].

pub mod dataflow_ir;
pub mod progir;
pub mod sentient;
