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

//! `SinkScalarCopy.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e209_region_` | 209 | 0 | 3 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:89` |
//! | `e210_isUniformLocalRegion` | 210 | 0 | 4 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:94` |
//! | `e211_dump` | 211 | 0 | 5 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:100` |
//! | `e212_op_` | 212 | 0 | 3 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:116` |
//! | `e213_addRegion` | 213 | 0 | 12 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:125` |
//! | `e214_sinkCopyOps` | 214 | 0 | 16 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:215` |
//! | `e215_clear` | 215 | 0 | 5 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:232` |
//! | `e381_dump` | 381 | 1 | 7 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:144` |
//! | `e477_runOn` | 477 | 2 | 25 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:189` |
//! | `e538_runOn` | 538 | 3 | 7 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:176` |
//! | `e539_runOn` | 539 | 3 | 4 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:184` |
//! | `e584_runOnOperation` | 584 | 4 | 7 | `dcc/src/Transform/Sentient/SinkScalarCopy.cpp:167` |


// crustify:todo: e209_region_
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:89  (3 body lines, level 0)
//   original  : RegionInfo(Region *region) : region_(region)

// crustify:todo: e210_isUniformLocalRegion
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:94  (4 body lines, level 0)
//   original  : bool isUniformLocalRegion() const

// crustify:todo: e211_dump
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:100  (5 body lines, level 0)
//   original  : void dump(raw_ostream &os, int indent = 0) const

// crustify:todo: e212_op_
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:116  (3 body lines, level 0)
//   original  : UsesInfo(Operation *op) : op_(op)

// crustify:todo: e213_addRegion
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:125  (12 body lines, level 0)
//   original  : void addRegion(Region *region)

// crustify:todo: e214_sinkCopyOps
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:215  (16 body lines, level 0)
//   original  : void sinkCopyOps()

// crustify:todo: e215_clear
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:232  (5 body lines, level 0)
//   original  : void clear()

// crustify:todo: e381_dump
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:144  (7 body lines, level 1)
//   original  : void dump(raw_ostream &os, int indent = 0) const
//   calls     : e211_dump

// crustify:todo: e477_runOn
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:189  (25 body lines, level 2)
//   original  : void runOn(sentient::CopyOp copy_op)
//   calls     : e211_dump, e213_addRegion, e381_dump

// crustify:todo: e538_runOn
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:176  (7 body lines, level 3)
//   original  : void runOn(dataflow::ProgramUnitOp unit)
//   calls     : e214_sinkCopyOps, e215_clear, e477_runOn, e539_runOn

// crustify:todo: e539_runOn
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:184  (4 body lines, level 3)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e477_runOn, e538_runOn

// crustify:todo: e584_runOnOperation
//   authority : dcc/src/Transform/Sentient/SinkScalarCopy.cpp:167  (7 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e477_runOn, e538_runOn, e539_runOn

