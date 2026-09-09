// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `SetSendDestinationRE.cpp` — 4 of the campaign's 656 units (dependency level(s) [0, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e195_determineOptimizationMode` | 195 | 0 | 22 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:184` |
//! | `e475_runOn` | 475 | 2 | 54 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:109` |
//! | `e537_runOn` | 537 | 3 | 4 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:164` |
//! | `e583_runOnOperation` | 583 | 4 | 5 | `dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:169` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so the two GenValue families below are reachable
// only from each other and their tests until `e583_runOnOperation` (level 4) lands. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e583: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::uniform;
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::Val;
use crate::units::DfirUnit;

pub(crate) mod composite_set_dst_gen_value_lxlu;
pub(crate) mod composite_set_dst_gen_value_sfp;
pub(crate) mod set_dst_gen_value_lxlu;
pub(crate) mod set_dst_gen_value_sfp;
pub(crate) mod set_send_dst_rde_tree;
pub(crate) mod simple_set_dst_gen_value_lxlu;
pub(crate) mod simple_set_dst_gen_value_sfp;

/// `SetDestREOptimizationMode` (`SetSendDestinationRE.hpp:23-27`) — WHICH SETTING THIS PROGRAM UNIT'S
/// redundant `set_send_dst`s are about: the LXLU's `SETDSTMASK` or the SFP's `SETDEST`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SetDestReOptimizationMode {
    /// `kUnknown` — the units are neither, and [`e195`](determine_optimization_mode)'s caller skips
    /// the unit entirely (`SetSendDestinationRE.cpp:110-120`).
    #[default]
    Unknown,
    /// `kOptimizeForLXLU`.
    OptimizeForLxlu,
    /// `kOptimizeForSFP`.
    OptimizeForSfp,
}

/// A `uniform::QueryMapOp` AS A COMPOSITE GEN VALUE HOLDS IT — the op the reference stores by handle,
/// compares by identity and streams whole.
///
/// ⭐ THE THREE OPERANDS, NOT AN `Op` — `uniform::QueryMapOp` is a TYPED handle, so a composite
/// GenValue cannot be holding any other op, and [`uniform::Op`] could be. Identity survives the
/// change because the value minter gives every op a distinct [`Self::result`]
/// ([`crate::islands::dataflow_ir::Values`]), so comparing results IS comparing handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryMapOp {
    result: Val,
    map: Val,
    key: Val,
}

impl QueryMapOp {
    /// The op that binds `result` by reading `map` at `key`.
    #[must_use]
    pub(crate) const fn of(result: Val, map: Val, key: Val) -> QueryMapOp {
        QueryMapOp { result, map, key }
    }

    /// `getMap()` — the `uniform.def_immutable_mapping` being read.
    #[must_use]
    pub(crate) const fn map(self) -> Val {
        self.map
    }

    /// `getKey()` — which unit's value to read.
    #[must_use]
    pub(crate) const fn key(self) -> Val {
        self.key
    }

    /// `OS << qmap_` — MLIR's `operator<<(raw_ostream &, OpState)`, which prints the whole operation.
    ///
    /// ⛔ NO TRAILING NEWLINE. [`uniform::emit`] writes one because it renders a line of a module;
    /// streaming an op into a debug dump does not, and the reference's callers add their own.
    pub(crate) fn print(self, out: &mut String) {
        let mut line = String::new();
        uniform::emit(
            &mut line,
            &uniform::Op::QueryMap {
                result: self.result,
                map: self.map,
                key: self.key,
            },
            0,
        );
        out.push_str(line.trim_end_matches('\n'));
    }
}

/// Replaces: e195_determineOptimizationMode
///
/// Which setting to optimize is decided by the kind of unit the program unit runs on.
///
/// ⛔⛔ `DT_CHECK_MSG(!(all_lxlu && all_sfp), "it cannot be that all units are both lxlu and sfp")`
/// IS A TYPE GUARD HERE, NOT A CHECK. [`crate::islands::dataflow_ir::Units`] is constructed BY kind
/// and is non-empty, so a unit list cannot be all-`lxlu` and all-`sfp` at once, and neither
/// `llvm::all_of` has anything left to scan — one `kind()` answers both.
///
/// ⭐ AND THE REFERENCE'S EMPTY-LIST CASE GOES WITH IT: two vacuous `all_of`s would both be true and
/// abort, which `Units` having a `head` makes unreachable rather than latent.
#[must_use]
pub(crate) fn determine_optimization_mode<A: Arch>(
    unit: &ProgramUnit<A>,
) -> SetDestReOptimizationMode {
    match unit.on.kind() {
        DfirUnit::Lxlu => SetDestReOptimizationMode::OptimizeForLxlu,
        DfirUnit::Sfp => SetDestReOptimizationMode::OptimizeForSfp,
        _ => SetDestReOptimizationMode::Unknown,
    }
}

// crustify:todo: e475_runOn
//   authority : dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:109  (54 body lines, level 2)
//   original  : void runOn(dataflow::ProgramUnitOp unit)
//   calls     : e195_determineOptimizationMode, e200_print, e201_print, e202_print, e205_print, e206_print, e207_print, e378_optimize

// crustify:todo: e537_runOn
//   authority : dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:164  (4 body lines, level 3)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e475_runOn

// crustify:todo: e583_runOnOperation
//   authority : dcc/src/Transform/Sentient/SetSendDestinationRE.cpp:169  (5 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e475_runOn, e537_runOn

#[cfg(test)]
mod unit_tests {
    use super::{SetDestReOptimizationMode, determine_optimization_mode};
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::sentient::ProgramUnit;
    use crate::islands::sentient::dialects::Val;
    use crate::units::DfirUnit;

    /// A program unit bound to one unit of `kind`.
    fn unit_on(kind: DfirUnit) -> ProgramUnit<Dd2> {
        ProgramUnit::<Dd2> {
            iter_arg: None,
            on: Units::one(kind, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        }
    }

    /// e195 — `lxlu` and `sfp` each pick their own setting and everything else is unknown.
    #[test]
    fn e195_reads_the_optimization_mode_off_the_unit_kind() {
        assert_eq!(
            determine_optimization_mode(&unit_on(DfirUnit::Lxlu)),
            SetDestReOptimizationMode::OptimizeForLxlu
        );
        assert_eq!(
            determine_optimization_mode(&unit_on(DfirUnit::Sfp)),
            SetDestReOptimizationMode::OptimizeForSfp
        );
        // `else opt_mode_ = kUnknown` — the caller then skips the unit.
        assert_eq!(
            determine_optimization_mode(&unit_on(DfirUnit::L3lu)),
            SetDestReOptimizationMode::Unknown
        );
    }
}
