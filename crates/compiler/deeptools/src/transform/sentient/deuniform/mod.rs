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

//! `Deuniform.cpp` — 9 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e039_existsInCollection` | 039 | 0 | 8 | `dcc/src/Transform/Sentient/Deuniform.cpp:128` |
//! | `e297_simplifyProgramUnitOp` | 297 | 1 | 64 | `dcc/src/Transform/Sentient/Deuniform.cpp:155` |
//! | `e298_duplicateAndUpdateEntriesOfQueryMap` | 298 | 1 | 22 | `dcc/src/Transform/Sentient/Deuniform.cpp:221` |
//! | `e299_expandAllGroupsToUnitsAndUpdateSizes` | 299 | 1 | 35 | `dcc/src/Transform/Sentient/Deuniform.cpp:246` |
//! | `e436_duplicateAndUpdateEntriesOfQueryMapInLocalRegions` | 436 | 2 | 13 | `dcc/src/Transform/Sentient/Deuniform.cpp:139` |
//! | `e497_duplicateAndUpdateRegionsOfLocalRegionOps` | 497 | 3 | 88 | `dcc/src/Transform/Sentient/Deuniform.cpp:284` |
//! | `e556_deuniform` | 556 | 4 | 70 | `dcc/src/Transform/Sentient/Deuniform.cpp:375` |
//! | `e595_deuniform` | 595 | 5 | 30 | `dcc/src/Transform/Sentient/Deuniform.cpp:448` |
//! | `e621_runOnOperation` | 621 | 6 | 42 | `dcc/src/Transform/Sentient/Deuniform.cpp:482` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e621_runOnOperation` (level 6) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of a 9-unit module fails the gate.
// ⭐ REMOVE THIS WITH e621: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::sentient::dialects::Val;

/// Replaces: e039_existsInCollection
///
/// Whether this unit is already a member of one of the collected sets of units.
///
/// TRAP: the identity is the `Val` the `dataflow.get_unit` BINDS — every `unit_op` this pass compares
/// arrives through `unit.getDefiningOp()` (`dcc/src/Transform/Sentient/Deuniform.cpp:509`), so equal
/// values are the same op.
pub fn exists_in_collection(unit_op: Val, collection: &[Vec<Val>]) -> bool {
    collection.iter().any(|set| set.contains(&unit_op))
}

// crustify:todo: e297_simplifyProgramUnitOp
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:155  (64 body lines, level 1)
//   original  : void DeuniformPass::simplifyProgramUnitOp( mlir::dataflow::ProgramUnitOp prog_unit_op)
//   calls     : e252_size

// crustify:todo: e298_duplicateAndUpdateEntriesOfQueryMap
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:221  (22 body lines, level 1)
//   original  : mlir::uniform::QueryMapOp DeuniformPass::duplicateAndUpdateEntriesOfQueryMap( mlir::uniform::QueryMapOp query_map_op, std::vector<mlir::Value> new_units, mlir::OpBuilder builder)
//   calls     : e252_size

// crustify:todo: e299_expandAllGroupsToUnitsAndUpdateSizes
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:246  (35 body lines, level 1)
//   original  : void DeuniformPass::expandAllGroupsToUnitsAndUpdateSizes( std::vector<Value> &units, mlir::ArrayAttr &list_sizes, mlir::OpBuilder builder)
//   calls     : e252_size

// crustify:todo: e436_duplicateAndUpdateEntriesOfQueryMapInLocalRegions
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:139  (13 body lines, level 2)
//   original  : void DeuniformPass::duplicateAndUpdateEntriesOfQueryMapInLocalRegions( mlir::Operation *local_region_op, std::vector<mlir::Value> &new_units, std::vector<mlir::Operation *> &to_be_deleted)
//   calls     : e298_duplicateAndUpdateEntriesOfQueryMap

// crustify:todo: e497_duplicateAndUpdateRegionsOfLocalRegionOps
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:284  (88 body lines, level 3)
//   original  : mlir::Operation *DeuniformPass::duplicateAndUpdateRegionsOfLocalRegionOps( mlir::Operation *local_op, std::vector<mlir::Value> &new_units, mlir::OpBuilder builder, std::vector<mlir::Operation *> &to_be_deleted)
//   calls     : e252_size, e299_expandAllGroupsToUnitsAndUpdateSizes, e436_duplicateAndUpdateEntriesOfQueryMapInLocalRegions

// crustify:todo: e556_deuniform
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:375  (70 body lines, level 4)
//   original  : mlir::dataflow::ProgramUnitOp DeuniformPass::deuniform( mlir::dataflow::ProgramUnitOp prog_unit, std::vector<mlir::dataflow::GetUnitOp> &set_of_units)
//   calls     : e252_size, e298_duplicateAndUpdateEntriesOfQueryMap, e497_duplicateAndUpdateRegionsOfLocalRegionOps

// crustify:todo: e595_deuniform
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:448  (30 body lines, level 5)
//   original  : std::vector<mlir::dataflow::ProgramUnitOp> DeuniformPass::deuniform( mlir::dataflow::ProgramUnitOp prog_unit, std::vector<std::vector<mlir::dataflow::GetUnitOp>> &set_of_unit_collection)
//   calls     : e252_size, e556_deuniform

// crustify:todo: e621_runOnOperation
//   authority : dcc/src/Transform/Sentient/Deuniform.cpp:482  (42 body lines, level 6)
//   original  : void DeuniformPass::runOnOperation()
//   calls     : e039_existsInCollection, e252_size, e297_simplifyProgramUnitOp, e556_deuniform, e595_deuniform

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// A unit in the second set is found; one in no set is not.
    #[test]
    fn exists_in_collection_searches_every_set() {
        let collection = vec![vec![Val(1), Val(2)], vec![Val(3)]];

        assert!(exists_in_collection(Val(3), &collection));
        assert!(!exists_in_collection(Val(4), &collection));
    }
}
