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

//! `StoreAndForwardFusion.cpp` — 11 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e223_appendToVector` | 223 | 0 | 4 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:94` |
//! | `e224_findFusibleOp` | 224 | 0 | 29 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:117` |
//! | `e225_isOpTrivialMacOp` | 225 | 0 | 31 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:150` |
//! | `e226_isOpTrivialBinaryOp` | 226 | 0 | 21 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:183` |
//! | `e227_fillOperandForFusibleOp` | 227 | 0 | 27 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:350` |
//! | `e228_updateFusibleOp` | 228 | 0 | 41 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:387` |
//! | `e385_appendValueForwarding` | 385 | 1 | 6 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:100` |
//! | `e386_isEligibleToFuse` | 386 | 1 | 87 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:219` |
//! | `e387_fillOperandForTrivialOp` | 387 | 1 | 19 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:318` |
//! | `e480_FuseStoreAndForward` | 480 | 2 | 119 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:434` |
//! | `e541_runOnOperation` | 541 | 3 | 24 | `dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:555` |


// crustify:todo: e223_appendToVector
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:94  (4 body lines, level 0)
//   original  : static void appendToVector(std::vector<mlir::Attribute> &vector_attr, mlir::ArrayAttr &array_attr)

// crustify:todo: e224_findFusibleOp
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:117  (29 body lines, level 0)
//   original  : mlir::Operation *StoreAndForwardFusionPass::findFusibleOp( mlir::Operation *op, std::vector<mlir::Operation *> &to_be_deleted, bool is_fold_AB_mode)

// crustify:todo: e225_isOpTrivialMacOp
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:150  (31 body lines, level 0)
//   original  : bool StoreAndForwardFusionPass::isOpTrivialMacOp(sentient::MacOp mac_op)

// crustify:todo: e226_isOpTrivialBinaryOp
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:183  (21 body lines, level 0)
//   original  : bool StoreAndForwardFusionPass::isOpTrivialBinaryOp( sentient::BinaryOp binary_op)

// crustify:todo: e227_fillOperandForFusibleOp
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:350  (27 body lines, level 0)
//   original  : LogicalResult StoreAndForwardFusionPass::fillOperandForFusibleOp( mlir::Operation *fusible_op, mlir::ArrayAttr &opA_forwarding, mlir::ArrayAttr &opB_forwarding, mlir::ArrayAttr &opC_forwarding, mlir::ArrayAttr &result_forwarding, SentientPrecision &result_precision)

// crustify:todo: e228_updateFusibleOp
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:387  (41 body lines, level 0)
//   original  : LogicalResult StoreAndForwardFusionPass::updateFusibleOp( mlir::Operation *fusible_op, std::vector<mlir::Attribute> &value_forwardings_for_trivial)

// crustify:todo: e385_appendValueForwarding
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:100  (6 body lines, level 1)
//   original  : static void appendValueForwarding( std::vector<mlir::Attribute> &value_forwardings, mlir::ArrayAttr &opA_forwarding, mlir::ArrayAttr &opB_forwarding, mlir::ArrayAttr &opC_forwarding, mlir::ArrayAttr &result_forwarding)
//   calls     : e223_appendToVector

// crustify:todo: e386_isEligibleToFuse
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:219  (87 body lines, level 1)
//   original  : bool StoreAndForwardFusionPass::isEligibleToFuse( std::vector<mlir::Attribute> &value_forwardings_for_trivial, std::vector<mlir::Attribute> &value_forwardings_for_fusible, mlir::Operation *trivial_op, mlir::Operation *fusible_op)
//   calls     : e248_getFoldModeAttributeIfExists

// crustify:todo: e387_fillOperandForTrivialOp
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:318  (19 body lines, level 1)
//   original  : bool StoreAndForwardFusionPass::fillOperandForTrivialOp( mlir::Operation *op, mlir::ArrayAttr &opA_forwarding, mlir::ArrayAttr &opB_forwarding, mlir::ArrayAttr &opC_forwarding, mlir::ArrayAttr &result_forwarding, SentientPrecision &result_precision)
//   calls     : e225_isOpTrivialMacOp, e226_isOpTrivialBinaryOp

// crustify:todo: e480_FuseStoreAndForward
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:434  (119 body lines, level 2)
//   original  : LogicalResult StoreAndForwardFusionPass::FuseStoreAndForward( mlir::Operation *op, std::vector<mlir::Operation *> &to_be_deleted)
//   calls     : e223_appendToVector, e224_findFusibleOp, e227_fillOperandForFusibleOp, e228_updateFusibleOp, e248_getFoldModeAttributeIfExists, e385_appendValueForwarding, e386_isEligibleToFuse, e387_fillOperandForTrivialOp

// crustify:todo: e541_runOnOperation
//   authority : dcc/src/Transform/Sentient/StoreAndForwardFusion.cpp:555  (24 body lines, level 3)
//   original  : void StoreAndForwardFusionPass::runOnOperation()
//   calls     : e480_FuseStoreAndForward

