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

//! `LoopSplittingAndUnrolling.cpp` — 19 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e085_replaceIterArgsAndYieldResults` | 085 | 0 | 13 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:132` |
//! | `e086_isOkToUnroll` | 086 | 0 | 47 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:159` |
//! | `e087_cleanupAndRecalculate` | 087 | 0 | 37 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:295` |
//! | `e088_ifOpUsesIV` | 088 | 0 | 10 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:455` |
//! | `e089_hasIfOps` | 089 | 0 | 12 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:466` |
//! | `e090_evaluatePredicate` | 090 | 0 | 33 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:552` |
//! | `e091_findAllSplitVals` | 091 | 0 | 22 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:596` |
//! | `e092_computeReducedCost` | 092 | 0 | 39 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:649` |
//! | `e318_promoteForLoopBodyAndDelete` | 318 | 1 | 10 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:147` |
//! | `e319_getAllTrueIndexOfIfOpsList` | 319 | 1 | 7 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:587` |
//! | `e320_subsetsImpl` | 320 | 1 | 9 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:622` |
//! | `e446_subsets` | 446 | 2 | 12 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:635` |
//! | `e447_splitLoopNWay` | 447 | 2 | 129 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:858` |
//! | `e513_unrollLoop` | 513 | 3 | 86 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:207` |
//! | `e514_findOptimalNWaySplits` | 514 | 3 | 162 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:691` |
//! | `e566_getSavedCycleAndIbuffCost` | 566 | 4 | 117 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:335` |
//! | `e567_doSplitOrUnroll` | 567 | 4 | 23 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:993` |
//! | `e603_findBestLoopNodeToOptimize` | 603 | 5 | 69 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:479` |
//! | `e629_runOnOperation` | 629 | 6 | 61 | `dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:1019` |


// crustify:todo: e085_replaceIterArgsAndYieldResults
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:132  (13 body lines, level 0)
//   original  : void LoopSplittingAndUnrollingPass::replaceIterArgsAndYieldResults( sentient::ForOp for_op)

// crustify:todo: e086_isOkToUnroll
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:159  (47 body lines, level 0)
//   original  : bool LoopSplittingAndUnrollingPass::isOkToUnroll(sentient::ForOp for_op0)

// crustify:todo: e087_cleanupAndRecalculate
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:295  (37 body lines, level 0)
//   original  : void LoopSplittingAndUnrollingPass::cleanupAndRecalculate( dcc::LoopTree<sentient::ForOp, dataflow::ProgramUnitOp> &tree, InstructionEstimatorImpl &ie, dataflow::ProgramUnitOp &unit_op)

// crustify:todo: e088_ifOpUsesIV
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:455  (10 body lines, level 0)
//   original  : bool LoopSplittingAndUnrollingPass::ifOpUsesIV(dcc::LoopNode *n)

// crustify:todo: e089_hasIfOps
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:466  (12 body lines, level 0)
//   original  : bool LoopSplittingAndUnrollingPass::hasIfOps(dcc::LoopNode *n)

// crustify:todo: e090_evaluatePredicate
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:552  (33 body lines, level 0)
//   original  : bool LoopSplittingAndUnrollingPass::evaluatePredicate(int bound, sentient::IfOp &if_op)

// crustify:todo: e091_findAllSplitVals
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:596  (22 body lines, level 0)
//   original  : std::vector<std::pair<int, int>> LoopSplittingAndUnrollingPass::findAllSplitVals( std::map<int, std::vector<bool>, std::greater<>> &bound_to_all_ifops_predval)

// crustify:todo: e092_computeReducedCost
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:649  (39 body lines, level 0)
//   original  : void LoopSplittingAndUnrollingPass::computeReducedCost( dcc::CondNode *n, InstructionEstimatorImpl &ie, int &cost_reduced, std::map<sentient::IfOp, bool> &if_op_to_pred_val)

// crustify:todo: e318_promoteForLoopBodyAndDelete
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:147  (10 body lines, level 1)
//   original  : void LoopSplittingAndUnrollingPass::promoteForLoopBodyAndDelete( sentient::ForOp for_op)
//   calls     : e085_replaceIterArgsAndYieldResults

// crustify:todo: e319_getAllTrueIndexOfIfOpsList
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:587  (7 body lines, level 1)
//   original  : std::vector<bool> LoopSplittingAndUnrollingPass::getAllTrueIndexOfIfOpsList( int bound, llvm::SmallVectorImpl<sentient::IfOp> &if_list)
//   calls     : e090_evaluatePredicate

// crustify:todo: e320_subsetsImpl
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:622  (9 body lines, level 1)
//   original  : static void subsetsImpl(std::vector<std::pair<int, int>> &split_vals, std::vector<std::vector<std::pair<int, int>>> &res, std::vector<std::pair<int, int>> &subset, int index)
//   calls     : e252_size

// crustify:todo: e446_subsets
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:635  (12 body lines, level 2)
//   original  : static std::vector<std::vector<std::pair<int, int>>> subsets( std::vector<std::pair<int, int>> &split_vals)
//   calls     : e252_size, e320_subsetsImpl

// crustify:todo: e447_splitLoopNWay
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:858  (129 body lines, level 2)
//   original  : LogicalResult LoopSplittingAndUnrollingPass::splitLoopNWay( dcc::LoopNode *n, OpBuilder &const_builder)
//   calls     : e091_findAllSplitVals, e252_size, e319_getAllTrueIndexOfIfOpsList

// crustify:todo: e513_unrollLoop
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:207  (86 body lines, level 3)
//   original  : LogicalResult LoopSplittingAndUnrollingPass::unrollLoop( dcc::LoopNode *n, OpBuilder &const_builder)
//   calls     : e086_isOkToUnroll, e252_size, e318_promoteForLoopBodyAndDelete, e423_lookup

// crustify:todo: e514_findOptimalNWaySplits
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:691  (162 body lines, level 3)
//   original  : std::tuple<int, int, int, int> LoopSplittingAndUnrollingPass::findOptimalNWaySplits( dcc::LoopNode *n, InstructionEstimatorImpl &ie, dataflow::ProgramUnitOp &unit_op)
//   calls     : e091_findAllSplitVals, e092_computeReducedCost, e252_size, e319_getAllTrueIndexOfIfOpsList, e422_insert

// crustify:todo: e566_getSavedCycleAndIbuffCost
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:335  (117 body lines, level 4)
//   original  : std::pair<int, int> LoopSplittingAndUnrollingPass::getSavedCycleAndIbuffCost( dcc::LoopNode *loop, InstructionEstimatorImpl &ie, dataflow::ProgramUnitOp &unit, std::pair<int, int> &is_splitting)
//   calls     : e088_ifOpUsesIV, e514_findOptimalNWaySplits

// crustify:todo: e567_doSplitOrUnroll
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:993  (23 body lines, level 4)
//   original  : void LoopSplittingAndUnrollingPass::doSplitOrUnroll( OpBuilder &builder, dcc::LoopNode *n, std::pair<int, int> index_and_nwaysplit)
//   calls     : e447_splitLoopNWay, e513_unrollLoop

// crustify:todo: e603_findBestLoopNodeToOptimize
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:479  (69 body lines, level 5)
//   original  : std::pair<dcc::LoopNode *, std::pair<int, int>> LoopSplittingAndUnrollingPass::findBestLoopNodeToOptimize( InstructionEstimatorImpl &ie, dataflow::ProgramUnitOp &unit_op, bool disable_unrolling)
//   calls     : e088_ifOpUsesIV, e089_hasIfOps, e566_getSavedCycleAndIbuffCost

// crustify:todo: e629_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopSplittingAndUnrolling.cpp:1019  (61 body lines, level 6)
//   original  : void LoopSplittingAndUnrollingPass::runOnOperation()
//   calls     : e087_cleanupAndRecalculate, e567_doSplitOrUnroll, e597_runLightWeightSimplifications, e603_findBestLoopNodeToOptimize

