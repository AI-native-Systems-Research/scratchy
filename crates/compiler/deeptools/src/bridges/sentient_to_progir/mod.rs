// SPDX-License-Identifier: Apache-2.0
//! `SentientIR -> ProgIR` — dcc pass **D76**, the last MLIR rung, as one lowering over the TYPED
//! program.
//!
//! ```text
//! islands::sentient::Run<A, M, W>  ──here (D76)──►  islands::progir::Program<A, M, W>
//! ```
//!
//! # 🛑 THIS RUNG LEAVES MLIR
//!
//! ⛔⛔ **PROGIR IS NOT AN MLIR DIALECT.** `SentientToProgIR` fills in a plain C++ structure —
//! `std::map<int, ProgramAndStateInfo>` keyed by core id — and the MLIR module it leaves behind
//! holds only [`crate::islands::progir::dialects::init`]'s reference to it. So this bridge produces
//! a VALUE, and the only text it emits is the twelve-line `init.smc` module.
//!
//! ⭐ AND THE OUTPUT IS TINY. `kMaxCompIBuff = 256`, `kMaxCompRegs = 128`; a complete int8 batched
//! matmul compiles to **eight** instructions on one unit.
//!
//! # 🛑 WHAT IS AND IS NOT IN THIS SPAN
//!
//! ⭐ IN SCOPE: the 130 functions of `dcc/src/Conversion/SentientToProgIR/` (6879 declaration
//! lines), listed with their authority citations in `crustify-bridge3/UNITS.tsv` and their
//! exclusions in `crustify-bridge3/EXCLUSIONS.tsv`.
//!
//! ⛔⛔ **NOT IN SCOPE: `dcc/src/Transform/Sentient/`** — 32,766 lines of D29-D75 passes that
//! rewrite SentientIR *in place* (`RegisterAllocation`, `SmartRegisterAllocation`,
//! `RegisterTypeAssignment`, `AddressPinningAndToggle`, `LiveRangeReduction`, `OpRerolling`,
//! `LoopRolling`, …). They are island-internal transforms, not this conversion. A function here
//! that depends on state they establish — register assignments, pinned addresses, rerolled loops —
//! is ported **as the reference writes it**, with the dependency recorded in the commit message.
//! Do not port the pass, and do not invent the state.
//!
//! Ported from the authority tree `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`;
//! every function carries a `/// Replaces: eNNN_name` anchor citing its original.


/// `construct` — 46 units.
pub mod construct;

/// `lower` — 34 units.
pub mod lower;

/// `uniform` — 33 units.
pub mod uniform;

/// THE D76 PASS ITSELF — `runOnOperation`, `GenerateProgIR`, `GenerateProgIRForProgramUnit`,
pub mod driver;

/// WHICH REGISTERS ARE DEFINED, AND WHETHER ANYTHING READS ONE THAT IS NOT.
pub mod reg_def_tracker;

/// THE SHARED HELPERS — the on-the-fly conversion check, the proper-consumer walk, the address
pub mod utils;
