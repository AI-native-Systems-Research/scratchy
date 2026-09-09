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

//! `ScalarCopyInsertionForSymbols.cpp` — 14 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e149_collectOpsOfInterest` | 149 | 0 | 18 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:217` |
//! | `e150_pessimizeLiveness` | 150 | 0 | 20 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:276` |
//! | `e151_insertCopyOpsForJCRCandidate` | 151 | 0 | 23 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:416` |
//! | `e152_clear` | 152 | 0 | 7 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:441` |
//! | `e153_getLocale` | 153 | 0 | 26 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:449` |
//! | `e154_propagateElementSizeToCopyOp` | 154 | 0 | 24 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:476` |
//! | `e155_dumpSymbolicLocales` | 155 | 0 | 10 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:502` |
//! | `e156_dumpCandidates` | 156 | 0 | 12 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:545` |
//! | `e358_collectCandidates` | 358 | 1 | 65 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:297` |
//! | `e359_insertCopyOpsForCandidates` | 359 | 1 | 49 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:365` |
//! | `e360_dumpSymbolUsageInfo` | 360 | 1 | 32 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:512` |
//! | `e527_collectSymbolUsage` | 527 | 3 | 38 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:236` |
//! | `e576_runOn` | 576 | 4 | 67 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:148` |
//! | `e610_runOnOperation` | 610 | 5 | 17 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:130` |


// crustify:todo: e149_collectOpsOfInterest
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:217  (18 body lines, level 0)
//   original  : void ScalarCopyInsertionForSymbolsPass::collectOpsOfInterest( dataflow::ProgramUnitOp unit)

// crustify:todo: e150_pessimizeLiveness
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:276  (20 body lines, level 0)
//   original  : void ScalarCopyInsertionForSymbolsPass::pessimizeLiveness(Liveness &liveness)

// crustify:todo: e151_insertCopyOpsForJCRCandidate
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:416  (23 body lines, level 0)
//   original  : void ScalarCopyInsertionForSymbolsPass::insertCopyOpsForJCRCandidate( dataflow::ProgramUnitOp unit, const CandidateEntryTy candidate) const

// crustify:todo: e152_clear
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:441  (7 body lines, level 0)
//   original  : void ScalarCopyInsertionForSymbolsPass::clear()

// crustify:todo: e153_getLocale
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:449  (26 body lines, level 0)
//   original  : SentientRegType ScalarCopyInsertionForSymbolsPass::getLocale(OpOperand &use)

// crustify:todo: e154_propagateElementSizeToCopyOp
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:476  (24 body lines, level 0)
//   original  : bool ScalarCopyInsertionForSymbolsPass::propagateElementSizeToCopyOp( OpOperand &opnd, sentient::CopyOp copy_op)

// crustify:todo: e155_dumpSymbolicLocales
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:502  (10 body lines, level 0)
//   original  : void ScalarCopyInsertionForSymbolsPass::dumpSymbolicLocales() const

// crustify:todo: e156_dumpCandidates
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:545  (12 body lines, level 0)
//   original  : void ScalarCopyInsertionForSymbolsPass::dumpCandidates( CandidateListTy &candidates) const

// crustify:todo: e358_collectCandidates
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:297  (65 body lines, level 1)
//   original  : void ScalarCopyInsertionForSymbolsPass::collectCandidates( dataflow::ProgramUnitOp unit, CandidateListTy &candidates, RegisterPressure &rp) const
//   calls     : e252_size

// crustify:todo: e359_insertCopyOpsForCandidates
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:365  (49 body lines, level 1)
//   original  : void ScalarCopyInsertionForSymbolsPass::insertCopyOpsForCandidates( dataflow::ProgramUnitOp unit, const CandidateListTy &candidates)
//   calls     : e151_insertCopyOpsForJCRCandidate, e154_propagateElementSizeToCopyOp

// crustify:todo: e360_dumpSymbolUsageInfo
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:512  (32 body lines, level 1)
//   original  : void ScalarCopyInsertionForSymbolsPass::dumpSymbolUsageInfo() const
//   calls     : e252_size

// crustify:todo: e527_collectSymbolUsage
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:236  (38 body lines, level 3)
//   original  : void ScalarCopyInsertionForSymbolsPass::collectSymbolUsage( dataflow::ProgramUnitOp unit)
//   calls     : e153_getLocale, e422_insert

// crustify:todo: e576_runOn
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:148  (67 body lines, level 4)
//   original  : void ScalarCopyInsertionForSymbolsPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e149_collectOpsOfInterest, e150_pessimizeLiveness, e155_dumpSymbolicLocales, e156_dumpCandidates, e252_size, e358_collectCandidates, e359_insertCopyOpsForCandidates, e360_dumpSymbolUsageInfo, e527_collectSymbolUsage

// crustify:todo: e610_runOnOperation
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:130  (17 body lines, level 5)
//   original  : void ScalarCopyInsertionForSymbolsPass::runOnOperation()
//   calls     : e152_clear, e576_runOn

