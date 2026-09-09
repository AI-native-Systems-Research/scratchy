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

//! `ScalarOpReordering.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e174_ScalarOpReorderingPass` | 174 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:66` |
//! | `e175_runOn` | 175 | 0 | 9 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:79` |
//! | `e176_computeOpIndexing` | 176 | 0 | 8 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:98` |
//! | `e177_getLiverangeEndPt` | 177 | 0 | 9 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:109` |
//! | `e178_findAncestorInBlock` | 178 | 0 | 7 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:374` |
//! | `e367_runOnOperation` | 367 | 1 | 5 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:74` |
//! | `e368_getFirstUseWithinBlock` | 368 | 1 | 23 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:383` |
//! | `e369_getLastUseWithinBlock` | 369 | 1 | 23 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:407` |
//! | `e370_localeHasFreeRegs` | 370 | 1 | 30 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:431` |
//! | `e470_isCandidateForReordering` | 470 | 2 | 67 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:462` |
//! | `e531_findAndProcessCandidates` | 531 | 3 | 165 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:207` |
//! | `e579_runOn` | 579 | 4 | 15 | `dcc/src/Transform/Sentient/ScalarOpReordering.cpp:191` |


// crustify:todo: e174_ScalarOpReorderingPass
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:66  (4 body lines, level 0)
//   original  : explicit ScalarOpReorderingPass(const dcc::DccExtContext &dcc_ext_ctx, dcc::CommonPassOptions const &opts) : dcc_ext_ctx_(dcc_ext_ctx), opts_(opts), locale_has_free_regs_(kMaxNumLocales)

// crustify:todo: e175_runOn
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:79  (9 body lines, level 0)
//   original  : void runOn(ModuleOp module_op)

// crustify:todo: e176_computeOpIndexing
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:98  (8 body lines, level 0)
//   original  : void computeOpIndexing(dataflow::ProgramUnitOp unit_op)

// crustify:todo: e177_getLiverangeEndPt
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:109  (9 body lines, level 0)
//   original  : unsigned getLiverangeEndPt(Operation *op)

// crustify:todo: e178_findAncestorInBlock
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:374  (7 body lines, level 0)
//   original  : Operation *ScalarOpReorderingPass::findAncestorInBlock(Operation *op, Block *bb)

// crustify:todo: e367_runOnOperation
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:74  (5 body lines, level 1)
//   original  : void runOnOperation()
//   calls     : e175_runOn

// crustify:todo: e368_getFirstUseWithinBlock
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:383  (23 body lines, level 1)
//   original  : Operation *ScalarOpReorderingPass::getFirstUseWithinBlock(Value v)
//   calls     : e178_findAncestorInBlock

// crustify:todo: e369_getLastUseWithinBlock
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:407  (23 body lines, level 1)
//   original  : Operation *ScalarOpReorderingPass::getLastUseWithinBlock(Value v, Block *bb)
//   calls     : e178_findAncestorInBlock

// crustify:todo: e370_localeHasFreeRegs
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:431  (30 body lines, level 1)
//   original  : bool ScalarOpReorderingPass::localeHasFreeRegs(SentientRegType locale)
//   calls     : e056_set, e063_getMaxRegNum

// crustify:todo: e470_isCandidateForReordering
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:462  (67 body lines, level 2)
//   original  : bool ScalarOpReorderingPass::isCandidateForReordering(Operation &op)
//   calls     : e368_getFirstUseWithinBlock, e369_getLastUseWithinBlock

// crustify:todo: e531_findAndProcessCandidates
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:207  (165 body lines, level 3)
//   original  : void ScalarOpReorderingPass::findAndProcessCandidates( dataflow::ProgramUnitOp unit_op)
//   calls     : e177_getLiverangeEndPt, e368_getFirstUseWithinBlock, e370_localeHasFreeRegs, e422_insert, e470_isCandidateForReordering

// crustify:todo: e579_runOn
//   authority : dcc/src/Transform/Sentient/ScalarOpReordering.cpp:191  (15 body lines, level 4)
//   original  : void ScalarOpReorderingPass::runOn(dataflow::ProgramUnitOp unit_op)
//   calls     : e176_computeOpIndexing, e531_findAndProcessCandidates

