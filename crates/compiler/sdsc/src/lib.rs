// SPDX-License-Identifier: Apache-2.0
//! # scratchy-sdsc — the staged SDSC lowering
//!
//! The SuperDSC (SDSC) backend lowers a `SubtileIR` decode tape to on-card device ops. Doing it in
//! one giant leap made it un-verifiable and bug-prone (unlike KTIR / SenGraph, which are thin
//! lowerings off the SAME `SubtileIR` and worked in hours). This crate replaces that leap with a
//! ladder of **distinct intermediate IR types**, each one lowering pass closer to the SDSC wire:
//!
//! ```text
//! SubtileIR ──► MacroIR ──► PrimIR ──► TiledIR ──► DeviceIR ──► SDSC wire
//!   (eval_dag)   (macro-steps) (primitives) (32-core tiles) (sticking/addr)
//! ```
//!
//! RULES (load-bearing):
//!  1. Each IR is its OWN type with its OWN `eval` — never `SubtileIR` reuse.
//!  2. Each bridge `IRn → IRn+1` expands any one node into **at most 3** finer nodes. If a step needs
//!     more, introduce ANOTHER intermediate IR between them.
//!  3. The IRs ARE the production lowering (successive passes), not tests beside it. Each bridge is
//!     verified by `eval(IRn+1) == eval(IRn)` on a seed, **anchored on `scratchy_subtile::eval_dag`**
//!     (the golden KTIR + SenGraph both match), wired in as a BUILD-TIME assert so a broken pass fails
//!     `cargo build`, localized to that one tiny bridge.

pub mod addr_ir;
pub mod core_map;
pub mod device_ir;
pub mod emit;
pub mod flat_ir;
pub mod fold_ir;
pub mod macro_ir;
pub mod op_dims;
pub mod prim_ir;
pub mod retile;
pub mod seg_layout;
pub mod sm_ir;
pub mod systolic_ir;
pub mod tiled_ir;
pub mod wire_ir;

// Exhaustive (all-inputs-in-bounds) proofs of the addressing/layout logic. `#[cfg(kani)]` ⇒ no effect on
// normal builds; run with `cargo kani -p scratchy-sdsc`.
#[cfg(kani)]
pub mod kani_proofs;
