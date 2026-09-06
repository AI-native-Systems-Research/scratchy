// SPDX-License-Identifier: Apache-2.0
//! BRIDGE 2 — `DataflowIR -> SentientIR`, the D1-D28 span of the deeptools ladder.
//!
//! ```text
//! DataflowIR ──D1–D28──► SentientIR ──D29–D75──► ──D76──► ProgIR
//!  (island)     (here)     (island)
//! ```
//!
//! ⭐ THE ORDER IS `docs/bridge2-porting-order.md` — 490 definitions in 11 dependency levels, where
//! definitions at one level never call each other. The gate is
//! `tests/the_bridge_matches_the_reference.rs`, a differential against the reference's own SentientIR
//! for our own emitted DataflowIR, and D29 re-entry is byte-exact so the end-to-end
//! `init_binary` comparison is available too.
//!
//! ⛔ PORT THE RULE, NEVER THE PLUMBING. Between 26% and 83% of the C++ in this span is `OpBuilder`,
//! SSA iterators, `getOrCreate…` memoisers and nested `(core, corelet, component)` lookup maps — all of
//! it there because the C++ builds an IR at run time. `#[forward]` resolves the tape at expansion and
//! this crate emits from constants, so none of it is owed.
//!
//! ⛔⛔ AND NEVER REFUSE AT RUNTIME. `Err`, `Result` and `.ok_or` do not belong in a bridge; a refusal
//! stops before the oracle, so the sentence we needed is never produced. An `Option` for a genuine
//! ABSENCE is fine — a precision no operand supplies is absent, not an error.

pub mod precision;
/// What a load's value reaches, and which unit consumes it.
pub mod load_chain;
/// Which sentient op an upstream arith or scf op becomes.
pub mod scalar_ops;
/// How an LX load arranges a sub-stick transfer.
pub mod ldtype;
/// Unit classification and naming, as the lowering asks it.
pub mod units;
/// The shapes the lowering accepts.
pub mod legality;
/// Which program units collapse, and how destinations split.
pub mod grouping;
