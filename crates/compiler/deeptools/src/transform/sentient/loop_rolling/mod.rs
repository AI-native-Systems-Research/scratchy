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

//! `LoopRolling.cpp` — 18 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5, 6, 7, 8]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e077_isIncrementField` | 077 | 0 | 12 | `dcc/src/Transform/Sentient/LoopRolling.cpp:67` |
//! | `e078_setEffectiveStart` | 078 | 0 | 4 | `dcc/src/Transform/Sentient/LoopRolling.cpp:134` |
//! | `e079_checkOpUsage` | 079 | 0 | 11 | `dcc/src/Transform/Sentient/LoopRolling.cpp:144` |
//! | `e080_getOperandKind` | 080 | 0 | 6 | `dcc/src/Transform/Sentient/LoopRolling.cpp:189` |
//! | `e081_deleteMatchedOps` | 081 | 0 | 4 | `dcc/src/Transform/Sentient/LoopRolling.cpp:233` |
//! | `e082_updateEndOpsOfMatchedOps` | 082 | 0 | 7 | `dcc/src/Transform/Sentient/LoopRolling.cpp:240` |
//! | `e083_computeValsIfDifferent` | 083 | 0 | 62 | `dcc/src/Transform/Sentient/LoopRolling.cpp:254` |
//! | `e084_updateBody` | 084 | 0 | 70 | `dcc/src/Transform/Sentient/LoopRolling.cpp:684` |
//! | `e317_checkDeltasOfOperands` | 317 | 1 | 79 | `dcc/src/Transform/Sentient/LoopRolling.cpp:505` |
//! | `e509_insertNextInstr` | 509 | 3 | 15 | `dcc/src/Transform/Sentient/LoopRolling.cpp:114` |
//! | `e510_insertOperandKind` | 510 | 3 | 3 | `dcc/src/Transform/Sentient/LoopRolling.cpp:183` |
//! | `e511_insertOperandDelta` | 511 | 3 | 3 | `dcc/src/Transform/Sentient/LoopRolling.cpp:186` |
//! | `e512_collectStartingValIterArgs` | 512 | 3 | 86 | `dcc/src/Transform/Sentient/LoopRolling.cpp:592` |
//! | `e565_collectDeltasOfOperands` | 565 | 4 | 109 | `dcc/src/Transform/Sentient/LoopRolling.cpp:392` |
//! | `e602_compareToNextWindow` | 602 | 5 | 67 | `dcc/src/Transform/Sentient/LoopRolling.cpp:321` |
//! | `e628_matchAndRoll` | 628 | 6 | 104 | `dcc/src/Transform/Sentient/LoopRolling.cpp:758` |
//! | `e642_rollInstrsInBlock` | 642 | 7 | 77 | `dcc/src/Transform/Sentient/LoopRolling.cpp:886` |
//! | `e652_runOnOperation` | 652 | 8 | 53 | `dcc/src/Transform/Sentient/LoopRolling.cpp:966` |


// crustify:todo: e077_isIncrementField
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:67  (12 body lines, level 0)
//   original  : bool isIncrementField(Operation &op, unsigned i, OpBuilder &const_builder)

// crustify:todo: e078_setEffectiveStart
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:134  (4 body lines, level 0)
//   original  : void setEffectiveStart(Block::iterator &effective_start, unsigned size)

// crustify:todo: e079_checkOpUsage
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:144  (11 body lines, level 0)
//   original  : bool checkOpUsage(Operation &op, Window *next_window)

// crustify:todo: e080_getOperandKind
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:189  (6 body lines, level 0)
//   original  : OperandKind getOperandKind(unsigned i)

// crustify:todo: e081_deleteMatchedOps
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:233  (4 body lines, level 0)
//   original  : void deleteMatchedOps()

// crustify:todo: e082_updateEndOpsOfMatchedOps
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:240  (7 body lines, level 0)
//   original  : void updateEndOpsOfMatchedOps(Window *end_window)

// crustify:todo: e083_computeValsIfDifferent
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:254  (62 body lines, level 0)
//   original  : bool computeValsIfDifferent(Value a_operand, Value b_operand, int64_t &delta)

// crustify:todo: e084_updateBody
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:684  (70 body lines, level 0)
//   original  : void updateBody(Window *start_window, Window *cur_window, Operation *yield_op, ForOp &for_op, OpBuilder &builder)

// crustify:todo: e317_checkDeltasOfOperands
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:505  (79 body lines, level 1)
//   original  : bool checkDeltasOfOperands(Window *cur_window, Operation &op_a, Operation &op_b, MatchedOp *matched_op)
//   calls     : e080_getOperandKind, e083_computeValsIfDifferent

// crustify:todo: e509_insertNextInstr
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:114  (15 body lines, level 3)
//   original  : void insertNextInstr(Block::iterator instr)
//   calls     : e422_insert

// crustify:todo: e510_insertOperandKind
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:183  (3 body lines, level 3)
//   original  : void insertOperandKind(unsigned i, OperandKind kind)
//   calls     : e422_insert

// crustify:todo: e511_insertOperandDelta
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:186  (3 body lines, level 3)
//   original  : void insertOperandDelta(unsigned i, int delta)
//   calls     : e422_insert

// crustify:todo: e512_collectStartingValIterArgs
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:592  (86 body lines, level 3)
//   original  : bool collectStartingValIterArgs(Window *start_window, Window *cur_window, SmallVector<Value, 2> &starting_vals, Value &zero)
//   calls     : e252_size, e422_insert

// crustify:todo: e565_collectDeltasOfOperands
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:392  (109 body lines, level 4)
//   original  : bool collectDeltasOfOperands(Window *start_window, Operation &op_a, Operation &op_b, MatchedOp *matched_op)
//   calls     : e077_isIncrementField, e083_computeValsIfDifferent, e510_insertOperandKind, e511_insertOperandDelta

// crustify:todo: e602_compareToNextWindow
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:321  (67 body lines, level 5)
//   original  : bool compareToNextWindow(bool is_start_window, Window *cur_window, Window *next_window)
//   calls     : e078_setEffectiveStart, e079_checkOpUsage, e252_size, e317_checkDeltasOfOperands, e565_collectDeltasOfOperands

// crustify:todo: e628_matchAndRoll
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:758  (104 body lines, level 6)
//   original  : void matchAndRoll(std::vector<Window *>::iterator &next)
//   calls     : e081_deleteMatchedOps, e082_updateEndOpsOfMatchedOps, e084_updateBody, e252_size, e512_collectStartingValIterArgs, e602_compareToNextWindow

// crustify:todo: e642_rollInstrsInBlock
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:886  (77 body lines, level 7)
//   original  : void rollInstrsInBlock(dcc::OperationEquivalence &oe, dcc::CommonPassOptions &opts, OpBuilder &const_builder, Block &bb, bool is_L3_case)
//   calls     : e509_insertNextInstr, e628_matchAndRoll

// crustify:todo: e652_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:966  (53 body lines, level 8)
//   original  : void runOnOperation()
//   calls     : e252_size, e642_rollInstrsInBlock

