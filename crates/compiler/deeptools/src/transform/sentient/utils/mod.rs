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

//! `Utils.cpp` — 19 of the campaign's 656 units (dependency level(s) [0, 1, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e239_getOperationOfBlock` | 239 | 0 | 8 | `dcc/src/Transform/Sentient/Utils.cpp:31` |
//! | `e240_selectIndicesForUnits` | 240 | 0 | 15 | `dcc/src/Transform/Sentient/Utils.cpp:66` |
//! | `e241_moveToCommonDominator` | 241 | 0 | 71 | `dcc/src/Transform/Sentient/Utils.cpp:84` |
//! | `e242_getForLoopInfoIfIV` | 242 | 0 | 37 | `dcc/src/Transform/Sentient/Utils.cpp:156` |
//! | `e243_roundDownUnrollFactor` | 243 | 0 | 32 | `dcc/src/Transform/Sentient/Utils.cpp:397` |
//! | `e244_negatePredicate` | 244 | 0 | 18 | `dcc/src/Transform/Sentient/Utils.cpp:431` |
//! | `e245_reversePredicate` | 245 | 0 | 18 | `dcc/src/Transform/Sentient/Utils.cpp:450` |
//! | `e246_getOutermostConstInitialization` | 246 | 0 | 55 | `dcc/src/Transform/Sentient/Utils.cpp:469` |
//! | `e247_memoryOpRequiresImmutAddrScalarCopy` | 247 | 0 | 64 | `dcc/src/Transform/Sentient/Utils.cpp:526` |
//! | `e248_getFoldModeAttributeIfExists` | 248 | 0 | 6 | `dcc/src/Transform/Sentient/Utils.cpp:593` |
//! | `e250_hasUniformizeRegion` | 250 | 0 | 12 | `dcc/src/Transform/Sentient/Utils.cpp:622` |
//! | `e251_add` | 251 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:165` |
//! | `e252_size` | 252 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:171` |
//! | `e253_areAllValuesEqual` | 253 | 0 | 6 | `dcc/src/Transform/Sentient/Utils.hpp:180` |
//! | `e392_getQueryKeyAndUnitsFromParentRegion` | 392 | 1 | 24 | `dcc/src/Transform/Sentient/Utils.cpp:40` |
//! | `e393_createSentientForOpWithAdditionalIterArgs` | 393 | 1 | 59 | `dcc/src/Transform/Sentient/Utils.cpp:195` |
//! | `e395_pruneOutOfScopeEntries` | 395 | 1 | 55 | `dcc/src/Transform/Sentient/Utils.cpp:635` |
//! | `e396_replaceValue` | 396 | 1 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:175` |
//! | `e546_findAndReplaceRedundantIterArgsUsedInConditions` | 546 | 3 | 138 | `dcc/src/Transform/Sentient/Utils.cpp:257` |

pub(crate) mod units_and_their_values;


// crustify:todo: e239_getOperationOfBlock
//   authority : dcc/src/Transform/Sentient/Utils.cpp:31  (8 body lines, level 0)
//   original  : Operation &getOperationOfBlock(Block &block, int op_num)

// crustify:todo: e240_selectIndicesForUnits
//   authority : dcc/src/Transform/Sentient/Utils.cpp:66  (15 body lines, level 0)
//   original  : void selectIndicesForUnits( SmallVector<Value> &units, SmallVector<unsigned> &indices, std::unordered_map<std::string, unsigned> unit_name_to_index_map)

// crustify:todo: e241_moveToCommonDominator
//   authority : dcc/src/Transform/Sentient/Utils.cpp:84  (71 body lines, level 0)
//   original  : mlir::LogicalResult moveToCommonDominator(Operation *op, Operation *new_use)

// crustify:todo: e242_getForLoopInfoIfIV
//   authority : dcc/src/Transform/Sentient/Utils.cpp:156  (37 body lines, level 0)
//   original  : std::tuple<Operation *, int64_t, int64_t, int64_t, int64_t> getForLoopInfoIfIV( Value val, bool allow_normalized_iv)

// crustify:todo: e243_roundDownUnrollFactor
//   authority : dcc/src/Transform/Sentient/Utils.cpp:397  (32 body lines, level 0)
//   original  : unsigned roundDownUnrollFactor(unsigned n, Operation *compute_op, SenTargets sen_target)

// crustify:todo: e244_negatePredicate
//   authority : dcc/src/Transform/Sentient/Utils.cpp:431  (18 body lines, level 0)
//   original  : CmpIPredicate negatePredicate(CmpIPredicate pred)

// crustify:todo: e245_reversePredicate
//   authority : dcc/src/Transform/Sentient/Utils.cpp:450  (18 body lines, level 0)
//   original  : CmpIPredicate reversePredicate(CmpIPredicate pred)

// crustify:todo: e246_getOutermostConstInitialization
//   authority : dcc/src/Transform/Sentient/Utils.cpp:469  (55 body lines, level 0)
//   original  : std::tuple<mlir::sentient::ForOp, int, int> getOutermostConstInitialization( BlockArgument iter_arg)

// crustify:todo: e247_memoryOpRequiresImmutAddrScalarCopy
//   authority : dcc/src/Transform/Sentient/Utils.cpp:526  (64 body lines, level 0)
//   original  : bool memoryOpRequiresImmutAddrScalarCopy(const dcc::DccExtContext &dcc_ext_ctx, SenComponents unit_type, Operation *op, bool do_range_check)

// crustify:todo: e248_getFoldModeAttributeIfExists
//   authority : dcc/src/Transform/Sentient/Utils.cpp:593  (6 body lines, level 0)
//   original  : std::optional<SentientFoldMode> getFoldModeAttributeIfExists(Operation *op)

// crustify:todo: e250_hasUniformizeRegion
//   authority : dcc/src/Transform/Sentient/Utils.cpp:622  (12 body lines, level 0)
//   original  : bool hasUniformizeRegion(dataflow::ProgramUnitOp unit)

// crustify:todo: e251_add
//   authority : dcc/src/Transform/Sentient/Utils.hpp:165  (4 body lines, level 0)
//   original  : void add(mlir::Value unit, mlir::Value value)

// crustify:todo: e252_size
//   authority : dcc/src/Transform/Sentient/Utils.hpp:171  (4 body lines, level 0)
//   original  : const size_t size() const

// crustify:todo: e253_areAllValuesEqual
//   authority : dcc/src/Transform/Sentient/Utils.hpp:180  (6 body lines, level 0)
//   original  : bool areAllValuesEqual()

// crustify:todo: e392_getQueryKeyAndUnitsFromParentRegion
//   authority : dcc/src/Transform/Sentient/Utils.cpp:40  (24 body lines, level 1)
//   original  : mlir::LogicalResult getQueryKeyAndUnitsFromParentRegion( Operation *op, Value &key, SmallVector<Value> &units)
//   calls     : e252_size

// crustify:todo: e393_createSentientForOpWithAdditionalIterArgs
//   authority : dcc/src/Transform/Sentient/Utils.cpp:195  (59 body lines, level 1)
//   original  : Operation *createSentientForOpWithAdditionalIterArgs( Operation *loop_op, std::vector<std::pair<Value, Value>> &start_vals_and_steps)
//   calls     : e252_size

// crustify:todo: e395_pruneOutOfScopeEntries
//   authority : dcc/src/Transform/Sentient/Utils.cpp:635  (55 body lines, level 1)
//   original  : void pruneOutOfScopeEntries(mlir::uniform::DefImmutableMappingOp map_op, SmallVectorImpl<Operation *> &to_be_deleted)
//   calls     : e252_size

// crustify:todo: e396_replaceValue
//   authority : dcc/src/Transform/Sentient/Utils.hpp:175  (4 body lines, level 1)
//   original  : void replaceValue(size_t index, mlir::Value new_val)
//   calls     : e252_size

// crustify:todo: e546_findAndReplaceRedundantIterArgsUsedInConditions
//   authority : dcc/src/Transform/Sentient/Utils.cpp:257  (138 body lines, level 3)
//   original  : LogicalResult findAndReplaceRedundantIterArgsUsedInConditions( mlir::sentient::ForOp loop, std::set<int> &iter_arg_indices_to_delete)
//   calls     : e422_insert

