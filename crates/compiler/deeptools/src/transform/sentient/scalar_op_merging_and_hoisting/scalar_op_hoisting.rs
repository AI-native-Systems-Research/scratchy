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

//! `ScalarOpMergingAndHoisting.cpp` — 13 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e170_addForOpResultAdjustment` | 170 | 0 | 28 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1543` |
//! | `e171_isMergeableOpOrChain` | 171 | 0 | 33 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1592` |
//! | `e172_addToOrReplaceOp` | 172 | 0 | 14 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1716` |
//! | `e173_hoistCandidateOutOfLoop` | 173 | 0 | 22 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1805` |
//! | `e365_applyOperationData` | 365 | 1 | 54 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1740` |
//! | `e468_processMergeableChain` | 468 | 2 | 58 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1645` |
//! | `e469_hoistForLinearChain` | 469 | 2 | 29 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2034` |
//! | `e529_processForOpResult` | 529 | 3 | 13 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1484` |
//! | `e530_processForDerivedIVElimination` | 530 | 3 | 38 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1843` |
//! | `e578_adjustCandidateForOpResult` | 578 | 4 | 19 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1514` |
//! | `e612_processForLinearChain` | 612 | 5 | 129 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1893` |
//! | `e613_processForGenericHoisting` | 613 | 5 | 81 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2075` |
//! | `e635_runScalarOpHoisting` | 635 | 6 | 61 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2184` |


// crustify:todo: e170_addForOpResultAdjustment
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1543  (28 body lines, level 0)
//   original  : void ScalarOpHoisting::addForOpResultAdjustment( sentient::ForOp *for_op, Value &result, const EvaluatedValue &adjustment_increment, int result_idx)

// crustify:todo: e171_isMergeableOpOrChain
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1592  (33 body lines, level 0)
//   original  : bool ScalarOpHoisting::isMergeableOpOrChain(Operation *input_to_chain, Operation *first_op_in_chain)

// crustify:todo: e172_addToOrReplaceOp
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1716  (14 body lines, level 0)
//   original  : Value ScalarOpHoisting::addToOrReplaceOp(Operation *op, const EvaluatedValue &modifier, bool replace_with_mod)

// crustify:todo: e173_hoistCandidateOutOfLoop
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1805  (22 body lines, level 0)
//   original  : void ScalarOpHoisting::hoistCandidateOutOfLoop(BlockArgument &main_iv, Operation *derived_iv, Value &new_add_offset)

// crustify:todo: e365_applyOperationData
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1740  (54 body lines, level 1)
//   original  : void ScalarOpHoisting::applyOperationData( SmallVector<OperationData> &ops_to_update)
//   calls     : e172_addToOrReplaceOp

// crustify:todo: e468_processMergeableChain
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1645  (58 body lines, level 2)
//   original  : bool ScalarOpHoisting::processMergeableChain( Operation *input_to_chain, Operation *first_op_in_chain, const EvaluatedValue &merging_increment, SmallVector<OperationData> &ops_to_update)
//   calls     : e158_doesValueExceedLRFRange, e361_isImmutableValueInRange

// crustify:todo: e469_hoistForLinearChain
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2034  (29 body lines, level 2)
//   original  : void ScalarOpHoisting::hoistForLinearChain( BlockArgument &main_iv, Operation *derived_iv, SmallVector<OperationData> &comp_ops)
//   calls     : e173_hoistCandidateOutOfLoop, e365_applyOperationData

// crustify:todo: e529_processForOpResult
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1484  (13 body lines, level 3)
//   original  : bool ScalarOpHoisting::processForOpResult( sentient::ForOp *for_op, int result_idx, const EvaluatedValue &adjustment_increment, SmallVector<OperationData> &ops_to_update)
//   calls     : e171_isMergeableOpOrChain, e468_processMergeableChain

// crustify:todo: e530_processForDerivedIVElimination
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1843  (38 body lines, level 3)
//   original  : bool ScalarOpHoisting::processForDerivedIVElimination(BlockArgument &main_iv, Operation *derived_iv)
//   calls     : e171_isMergeableOpOrChain, e365_applyOperationData, e468_processMergeableChain

// crustify:todo: e578_adjustCandidateForOpResult
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1514  (19 body lines, level 4)
//   original  : bool ScalarOpHoisting::adjustCandidateForOpResult( BlockArgument &main_iv, const EvaluatedValue &adjustment_increment)
//   calls     : e170_addForOpResultAdjustment, e365_applyOperationData, e529_processForOpResult

// crustify:todo: e612_processForLinearChain
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1893  (129 body lines, level 5)
//   original  : bool ScalarOpHoisting::processForLinearChain(BlockArgument &main_iv, Operation *derived_iv)
//   calls     : e361_isImmutableValueInRange, e469_hoistForLinearChain, e578_adjustCandidateForOpResult

// crustify:todo: e613_processForGenericHoisting
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2075  (81 body lines, level 5)
//   original  : bool ScalarOpHoisting::processForGenericHoisting(BlockArgument &main_iv, Operation *derived_iv)
//   calls     : e171_isMergeableOpOrChain, e173_hoistCandidateOutOfLoop, e365_applyOperationData, e578_adjustCandidateForOpResult

// crustify:todo: e635_runScalarOpHoisting
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2184  (61 body lines, level 6)
//   original  : void ScalarOpHoisting::runScalarOpHoisting()
//   calls     : e530_processForDerivedIVElimination, e612_processForLinearChain, e613_processForGenericHoisting

