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

//! `PortAssignment.cpp` — 13 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e123_clean` | 123 | 0 | 12 | `dcc/src/Transform/Sentient/PortAssignment.cpp:80` |
//! | `e124_getPortAttr` | 124 | 0 | 4 | `dcc/src/Transform/Sentient/PortAssignment.cpp:206` |
//! | `e125_getValidPorts` | 125 | 0 | 254 | `dcc/src/Transform/Sentient/PortAssignment.cpp:214` |
//! | `e126_IsSwappablePerAlgebraicReassociation` | 126 | 0 | 29 | `dcc/src/Transform/Sentient/PortAssignment.cpp:606` |
//! | `e339_addNodesToGraph` | 339 | 1 | 131 | `dcc/src/Transform/Sentient/PortAssignment.cpp:473` |
//! | `e340_updateSentientIRPorts` | 340 | 1 | 52 | `dcc/src/Transform/Sentient/PortAssignment.cpp:682` |
//! | `e341_addReuseToDummyOperands` | 341 | 1 | 36 | `dcc/src/Transform/Sentient/PortAssignment.cpp:739` |
//! | `e455_buildGraphNodes` | 455 | 2 | 6 | `dcc/src/Transform/Sentient/PortAssignment.cpp:636` |
//! | `e520_processDataID` | 520 | 3 | 67 | `dcc/src/Transform/Sentient/PortAssignment.cpp:93` |
//! | `e572_computePortLiveRange` | 572 | 4 | 40 | `dcc/src/Transform/Sentient/PortAssignment.cpp:165` |
//! | `e608_buildGraphEdges` | 608 | 5 | 37 | `dcc/src/Transform/Sentient/PortAssignment.cpp:644` |
//! | `e632_doPortAssignments` | 632 | 6 | 12 | `dcc/src/Transform/Sentient/PortAssignment.cpp:776` |
//! | `e645_runOnOperation` | 645 | 7 | 21 | `dcc/src/Transform/Sentient/PortAssignment.cpp:793` |


// crustify:todo: e123_clean
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:80  (12 body lines, level 0)
//   original  : void PortAssignmentPass::clean()

// crustify:todo: e124_getPortAttr
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:206  (4 body lines, level 0)
//   original  : mlir::IntegerAttr PortAssignmentPass::getPortAttr(Builder &builder, int opID)

// crustify:todo: e125_getValidPorts
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:214  (254 body lines, level 0)
//   original  : std::vector<int> PortAssignmentPass::getValidPorts(Operation *op, StringRef operand_value, const SenComponents comp, StringRef operand_precision, StringRef precision)

// crustify:todo: e126_IsSwappablePerAlgebraicReassociation
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:606  (29 body lines, level 0)
//   original  : int PortAssignmentPass::IsSwappablePerAlgebraicReassociation(Operation *op)

// crustify:todo: e339_addNodesToGraph
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:473  (131 body lines, level 1)
//   original  : void PortAssignmentPass::addNodesToGraph(Operation *op, const SenComponents comp)
//   calls     : e125_getValidPorts, e126_IsSwappablePerAlgebraicReassociation

// crustify:todo: e340_updateSentientIRPorts
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:682  (52 body lines, level 1)
//   original  : void PortAssignmentPass::updateSentientIRPorts(dataflow::ProgramUnitOp &unit)
//   calls     : e124_getPortAttr, e252_size

// crustify:todo: e341_addReuseToDummyOperands
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:739  (36 body lines, level 1)
//   original  : void PortAssignmentPass::addReuseToDummyOperands()
//   calls     : e124_getPortAttr, e252_size

// crustify:todo: e455_buildGraphNodes
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:636  (6 body lines, level 2)
//   original  : void PortAssignmentPass::buildGraphNodes(dataflow::ProgramUnitOp &unit, const SenComponents comp)
//   calls     : e339_addNodesToGraph

// crustify:todo: e520_processDataID
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:93  (67 body lines, level 3)
//   original  : void PortAssignmentPass::processDataID(int data_id, StringRef operand_value, Operation *current_op)
//   calls     : e422_insert

// crustify:todo: e572_computePortLiveRange
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:165  (40 body lines, level 4)
//   original  : void PortAssignmentPass::computePortLiveRange(dataflow::ProgramUnitOp &unit)
//   calls     : e252_size, e422_insert, e520_processDataID

// crustify:todo: e608_buildGraphEdges
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:644  (37 body lines, level 5)
//   original  : void PortAssignmentPass::buildGraphEdges(dataflow::ProgramUnitOp &unit)
//   calls     : e572_computePortLiveRange

// crustify:todo: e632_doPortAssignments
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:776  (12 body lines, level 6)
//   original  : void PortAssignmentPass::doPortAssignments(dataflow::ProgramUnitOp &unit, const SenComponents comp)
//   calls     : e123_clean, e340_updateSentientIRPorts, e341_addReuseToDummyOperands, e382_performGraphColoring, e455_buildGraphNodes, e608_buildGraphEdges

// crustify:todo: e645_runOnOperation
//   authority : dcc/src/Transform/Sentient/PortAssignment.cpp:793  (21 body lines, level 7)
//   original  : void PortAssignmentPass::runOnOperation()
//   calls     : e632_doPortAssignments

