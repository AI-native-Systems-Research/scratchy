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

//! `RegisterTypeAssignment.cpp` — 17 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e137_RegisterTypeAssignmentPass` | 137 | 0 | 6 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:77` |
//! | `e138_clear` | 138 | 0 | 6 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:222` |
//! | `e139_getLocale` | 139 | 0 | 8 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:231` |
//! | `e140_moveToCommonDominator` | 140 | 0 | 7 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:246` |
//! | `e141_addToWorkList` | 141 | 0 | 3 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:378` |
//! | `e142_isCopyNeeded` | 142 | 0 | 10 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:382` |
//! | `e143_updateAssignment` | 143 | 0 | 3 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:394` |
//! | `e144_printAssignments` | 144 | 0 | 14 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:425` |
//! | `e350_createCopyOperationAndUpdateAssignment` | 350 | 1 | 120 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:255` |
//! | `e351_addToWorkListAndUpdateAssignment` | 351 | 1 | 4 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:399` |
//! | `e352_updateProgramUnit` | 352 | 1 | 45 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:441` |
//! | `e463_addToWorkListCreateCopyAndUpdateAssignment` | 463 | 2 | 17 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:405` |
//! | `e523_processWorkList` | 523 | 3 | 93 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:492` |
//! | `e524_initializeAssignmentForAnOperation` | 524 | 3 | 383 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:594` |
//! | `e574_initializeWorkList` | 574 | 4 | 8 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:987` |
//! | `e609_runOn` | 609 | 5 | 55 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:996` |
//! | `e633_runOnOperation` | 633 | 6 | 17 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:113` |


// crustify:todo: e137_RegisterTypeAssignmentPass
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:77  (6 body lines, level 0)
//   original  : explicit RegisterTypeAssignmentPass(const dcc::DccExtContext& dcc_ext_ctx, dcc::CommonPassOptions const& opts, bool skip_copies) : dcc_ext_ctx_(dcc_ext_ctx), opts_(opts)

// crustify:todo: e138_clear
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:222  (6 body lines, level 0)
//   original  : void RegisterTypeAssignmentPass::clear()

// crustify:todo: e139_getLocale
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:231  (8 body lines, level 0)
//   original  : RegisterTypeAssignmentPass::RegisterLocales RegisterTypeAssignmentPass::getLocale(mlir::Value val) const

// crustify:todo: e140_moveToCommonDominator
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:246  (7 body lines, level 0)
//   original  : mlir::LogicalResult RegisterTypeAssignmentPass::moveToCommonDominator( Operation* op, Operation* new_use)

// crustify:todo: e141_addToWorkList
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:378  (3 body lines, level 0)
//   original  : void RegisterTypeAssignmentPass::addToWorkList(mlir::Value val)

// crustify:todo: e142_isCopyNeeded
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:382  (10 body lines, level 0)
//   original  : bool RegisterTypeAssignmentPass::isCopyNeeded( mlir::Value val, RegisterTypeAssignmentPass::RegisterLocales locale)

// crustify:todo: e143_updateAssignment
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:394  (3 body lines, level 0)
//   original  : void RegisterTypeAssignmentPass::updateAssignment( mlir::Value val, RegisterTypeAssignmentPass::RegisterLocales locale)

// crustify:todo: e144_printAssignments
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:425  (14 body lines, level 0)
//   original  : void RegisterTypeAssignmentPass::printAssignments()

// crustify:todo: e350_createCopyOperationAndUpdateAssignment
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:255  (120 body lines, level 1)
//   original  : Value RegisterTypeAssignmentPass::createCopyOperationAndUpdateAssignment( Value val, const RegisterLocales locale, Operation* user, const bool is_mutable_addr_or_xrf_ptr, int element_size)
//   calls     : e140_moveToCommonDominator

// crustify:todo: e351_addToWorkListAndUpdateAssignment
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:399  (4 body lines, level 1)
//   original  : void RegisterTypeAssignmentPass::addToWorkListAndUpdateAssignment( Value val, RegisterLocales locale)
//   calls     : e141_addToWorkList, e143_updateAssignment

// crustify:todo: e352_updateProgramUnit
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:441  (45 body lines, level 1)
//   original  : void RegisterTypeAssignmentPass::updateProgramUnit()
//   calls     : e252_size

// crustify:todo: e463_addToWorkListCreateCopyAndUpdateAssignment
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:405  (17 body lines, level 2)
//   original  : void RegisterTypeAssignmentPass::addToWorkListCreateCopyAndUpdateAssignment( MutableOperandRange operand, const RegisterLocales locale, Operation* user, int element_size, const bool is_mutable_addr_or_xrf_ptr)
//   calls     : e141_addToWorkList, e142_isCopyNeeded, e143_updateAssignment, e252_size, e350_createCopyOperationAndUpdateAssignment

// crustify:todo: e523_processWorkList
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:492  (93 body lines, level 3)
//   original  : void RegisterTypeAssignmentPass::processWorkList()
//   calls     : e139_getLocale, e143_updateAssignment, e351_addToWorkListAndUpdateAssignment, e463_addToWorkListCreateCopyAndUpdateAssignment

// crustify:todo: e524_initializeAssignmentForAnOperation
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:594  (383 body lines, level 3)
//   original  : void RegisterTypeAssignmentPass::initializeAssignmentForAnOperation( Operation* op)
//   calls     : e139_getLocale, e143_updateAssignment, e247_memoryOpRequiresImmutAddrScalarCopy, e252_size, e278_isValid, e351_addToWorkListAndUpdateAssignment, e463_addToWorkListCreateCopyAndUpdateAssignment

// crustify:todo: e574_initializeWorkList
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:987  (8 body lines, level 4)
//   original  : void RegisterTypeAssignmentPass::initializeWorkList()
//   calls     : e138_clear, e422_insert, e524_initializeAssignmentForAnOperation

// crustify:todo: e609_runOn
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:996  (55 body lines, level 5)
//   original  : void RegisterTypeAssignmentPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e352_updateProgramUnit, e523_processWorkList, e574_initializeWorkList

// crustify:todo: e633_runOnOperation
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:113  (17 body lines, level 6)
//   original  : void runOnOperation() override
//   calls     : e524_initializeAssignmentForAnOperation, e609_runOn

