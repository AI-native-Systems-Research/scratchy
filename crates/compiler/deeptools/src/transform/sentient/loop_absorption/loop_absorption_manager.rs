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

//! `LoopAbsorption.cpp` — 10 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e066_LoopAbsorptionManager` | 066 | 0 | 6 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:37` |
//! | `e067_updateLoopBound` | 067 | 0 | 48 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:58` |
//! | `e068_doChildrenLeftSiblingSubtreeHeightsMatch` | 068 | 0 | 12 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:122` |
//! | `e069_doChildrenRightSiblingSubtreeHeightsMatch` | 069 | 0 | 12 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:269` |
//! | `e070_removeNodesFromWorklistAndTree` | 070 | 0 | 7 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:410` |
//! | `e309_absorbIntoAnchorFromLeft` | 309 | 1 | 15 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:236` |
//! | `e310_absorbIntoAnchorFromRight` | 310 | 1 | 17 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:387` |
//! | `e506_canAbsorbToTheLeft` | 506 | 3 | 99 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:135` |
//! | `e507_canAbsorbToTheRight` | 507 | 3 | 103 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:282` |
//! | `e562_absorptionAnalysis` | 562 | 4 | 65 | `dcc/src/Transform/Sentient/LoopAbsorption.cpp:424` |


// crustify:todo: e066_LoopAbsorptionManager
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:37  (6 body lines, level 0)
//   original  : LoopAbsorptionManager::LoopAbsorptionManager( std::vector<dcc::LoopNode *> &sorted_worklist, dcc::LoopTree<sentient::ForOp, Operation *> &tree) : sorted_worklist_(sorted_worklist), tree_(tree), anchor_(sorted_worklist.back()), oe_(dcc::OperationEquivalence(nullptr, nullptr, DEBUG_TYPE, /*do_recursiv

// crustify:todo: e067_updateLoopBound
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:58  (48 body lines, level 0)
//   original  : void LoopAbsorptionManager::updateLoopBound(int64_t increment)

// crustify:todo: e068_doChildrenLeftSiblingSubtreeHeightsMatch
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:122  (12 body lines, level 0)
//   original  : bool LoopAbsorptionManager::doChildrenLeftSiblingSubtreeHeightsMatch()

// crustify:todo: e069_doChildrenRightSiblingSubtreeHeightsMatch
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:269  (12 body lines, level 0)
//   original  : bool LoopAbsorptionManager::doChildrenRightSiblingSubtreeHeightsMatch()

// crustify:todo: e070_removeNodesFromWorklistAndTree
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:410  (7 body lines, level 0)
//   original  : void LoopAbsorptionManager::removeNodesFromWorklistAndTree( const std::vector<dcc::LoopNode *> &nodes_to_delete)

// crustify:todo: e309_absorbIntoAnchorFromLeft
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:236  (15 body lines, level 1)
//   original  : void LoopAbsorptionManager::absorbIntoAnchorFromLeft()
//   calls     : e252_size

// crustify:todo: e310_absorbIntoAnchorFromRight
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:387  (17 body lines, level 1)
//   original  : void LoopAbsorptionManager::absorbIntoAnchorFromRight()
//   calls     : e252_size

// crustify:todo: e506_canAbsorbToTheLeft
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:135  (99 body lines, level 3)
//   original  : bool LoopAbsorptionManager::canAbsorbToTheLeft( std::vector<dcc::LoopNode *> &nodes_to_delete)
//   calls     : e068_doChildrenLeftSiblingSubtreeHeightsMatch, e422_insert

// crustify:todo: e507_canAbsorbToTheRight
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:282  (103 body lines, level 3)
//   original  : bool LoopAbsorptionManager::canAbsorbToTheRight( std::vector<dcc::LoopNode *> &nodes_to_delete)
//   calls     : e069_doChildrenRightSiblingSubtreeHeightsMatch, e422_insert

// crustify:todo: e562_absorptionAnalysis
//   authority : dcc/src/Transform/Sentient/LoopAbsorption.cpp:424  (65 body lines, level 4)
//   original  : void LoopAbsorptionManager::absorptionAnalysis()
//   calls     : e067_updateLoopBound, e070_removeNodesFromWorklistAndTree, e309_absorbIntoAnchorFromLeft, e310_absorbIntoAnchorFromRight, e506_canAbsorbToTheLeft, e507_canAbsorbToTheRight

