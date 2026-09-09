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

//! `ScalarOpMergingAndHoisting.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e157_doesImmutableImmExceedRange` | 157 | 0 | 18 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:145` |
//! | `e158_doesValueExceedLRFRange` | 158 | 0 | 21 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:167` |
//! | `e159_computeAddressScale` | 159 | 0 | 7 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:266` |
//! | `e160_MemoryOpInfo` | 160 | 0 | 38 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:288` |
//! | `e161_addSpeculativeImmutable` | 161 | 0 | 3 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:389` |
//! | `e162_addOpToBlock` | 162 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:429` |
//! | `e163_setEnablesMerging` | 163 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:434` |
//! | `e164_addUnrollCandidate` | 164 | 0 | 4 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:441` |
//! | `e361_isImmutableValueInRange` | 361 | 1 | 28 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:232` |
//! | `e366_runOn` | 366 | 1 | 130 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2258` |
//! | `e636_runOn` | 636 | 6 | 14 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2389` |
//! | `e646_runOnOperation` | 646 | 7 | 5 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2404` |

pub(crate) mod scalar_op_hoisting;
pub(crate) mod scalar_op_merging;


// crustify:todo: e157_doesImmutableImmExceedRange
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:145  (18 body lines, level 0)
//   original  : static bool doesImmutableImmExceedRange(const EvaluatedValue &immutable_addr_ev, int element_size, std::pair<int, int> ldsti_imm_range, int address_granularity_scale)

// crustify:todo: e158_doesValueExceedLRFRange
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:167  (21 body lines, level 0)
//   original  : static bool doesValueExceedLRFRange(const SenSystemDef &sys_def, const EvaluatedValue &value_ev, int element_size, SenComponents comp, int address_granularity_scale)

// crustify:todo: e159_computeAddressScale
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:266  (7 body lines, level 0)
//   original  : static int computeAddressScale(SenComponents comp, const dcc::DccExtContext &dcc_ext_context)

// crustify:todo: e160_MemoryOpInfo
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:288  (38 body lines, level 0)
//   original  : explicit MemoryOpInfo(Operation *op)

// crustify:todo: e161_addSpeculativeImmutable
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:389  (3 body lines, level 0)
//   original  : void addSpeculativeImmutable(const EvaluatedValue &speculative_immutable)

// crustify:todo: e162_addOpToBlock
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:429  (4 body lines, level 0)
//   original  : void addOpToBlock(OperationData op_data, bool is_add_or_sub = false)

// crustify:todo: e163_setEnablesMerging
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:434  (4 body lines, level 0)
//   original  : void setEnablesMerging()

// crustify:todo: e164_addUnrollCandidate
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:441  (4 body lines, level 0)
//   original  : void addUnrollCandidate(FieldUnrollData &unroll_candidate)

// crustify:todo: e361_isImmutableValueInRange
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:232  (28 body lines, level 1)
//   original  : bool isImmutableValueInRange(const EvaluatedValue &new_immutable_ev, const EvaluatedValue &original_immutable_ev, int element_size, mlir::Operation *op) const
//   calls     : e157_doesImmutableImmExceedRange, e158_doesValueExceedLRFRange

// crustify:todo: e366_runOn
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2258  (130 body lines, level 1)
//   original  : void runOn(dataflow::ProgramUnitOp unit, ModuleOp module_op)
//   calls     : e252_size

// crustify:todo: e636_runOn
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2389  (14 body lines, level 6)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e366_runOn, e597_runLightWeightSimplifications

// crustify:todo: e646_runOnOperation
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2404  (5 body lines, level 7)
//   original  : void runOnOperation()
//   calls     : e366_runOn, e636_runOn

