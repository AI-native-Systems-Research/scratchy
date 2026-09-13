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

//! `LiveRangeReduction.cpp` — 18 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e056_set` | 056 | 0 | 5 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:118` |
//! | `e057_print` | 057 | 0 | 14 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:141` |
//! | `e058_areElementSizeIdentical` | 058 | 0 | 9 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:466` |
//! | `e059_getFirstGlobalOrConstantAncestor` | 059 | 0 | 56 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1107` |
//! | `e060_areReglocalesMatching` | 060 | 0 | 5 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1261` |
//! | `e304_printSsaMap` | 304 | 1 | 18 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:218` |
//! | `e305_getBaseExpr` | 305 | 1 | 116 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:271` |
//! | `e306_getRootIterArg` | 306 | 1 | 72 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:393` |
//! | `e307_cloneValueToRegion` | 307 | 1 | 40 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:917` |
//! | `e440_addToMap` | 440 | 2 | 25 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:241` |
//! | `e441_findDominantValue` | 441 | 2 | 38 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:480` |
//! | `e442_addResultToYield` | 442 | 2 | 140 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:961` |
//! | `e443_createMapAndQuery` | 443 | 2 | 43 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1216` |
//! | `e502_reconstructOperation` | 502 | 3 | 259 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:546` |
//! | `e503_mapAllValues` | 503 | 3 | 48 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:827` |
//! | `e504_optimizeUniformRegionYieldedValues` | 504 | 3 | 47 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1167` |
//! | `e560_reduceLiveRange` | 560 | 4 | 28 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:885` |
//! | `e598_runOnOperation` | 598 | 5 | 45 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1267` |


// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e598_runOnOperation` (level 5) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaves of an 18-unit module fail the gate.
// ⭐ REMOVE THIS WITH e598: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use core::fmt::Write as _;
use std::collections::BTreeMap;

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::dataflow_ir::ty::{GenericComp, ScalarTy};
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::sentient::dialects;
use crate::islands::sentient::dialects::{
    Definitions, LocalRegion, Op, UniformRegions, Val, YieldedReg, element_size,
    erase_defining_op, sentient, symbol, uniform, value_reg_locale,
};
use crate::islands::sentient::{Program, print};
use crate::model::Model;
use crate::transform::sentient::analyses::{
    InstructionEstimator, OutOfScopeInstructionEstimator, OutOfScopeUnitIndexMap, UnitIndex,
    UnitIndexMap,
};
use crate::transform::sentient::utils::{self, OpAt};
use crate::workload::Workload;

/// `affine::FlatAffineValueConstraints` — THE LOCAL-VARIABLE CONSTRAINT SYSTEM OF ONE FLATTENED
/// AFFINE EXPRESSION, opaque here.
///
/// ⛔ MLIR UPSTREAM, AND NOTHING IN THIS CAMPAIGN BUILDS OR READS ONE. The only producer is
/// `mlir::getFlattenedAffineExpr` (`LiveRangeReduction.cpp:308-310`, out of campaign scope) and the only
/// readers are the system's own `isEqual` and `print`. So this is an IDENTITY, like
/// [`crate::transform::sentient::analyses::EvaluatedValue`]: which system, not the system.
///
/// ⛔ NO DERIVED `PartialEq`: `ExprInfo::operator==` compares two systems with `isEqual`, which is
/// CONTENT equality (`:135`). Comparing identities would answer a different question.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatAffineValueConstraints(pub u32);

impl FlatAffineValueConstraints {
    /// `FlatAffineValueConstraints::print` (`IntegerRelation::print`) — MLIR upstream's own dump of
    /// the constraint system, and out of campaign scope.
    ///
    /// ⛔ NOT INVENTABLE: it renders the system's equalities, inequalities and local-variable
    /// divisions, none of which this identity carries. Reached only from [`ExprInfo::print`], which is
    /// a `DEBUG_WITH_TYPE` trace.
    fn write(self, _out: &mut String) {
        todo!(
            "affine::FlatAffineValueConstraints::print (mlir/Analysis/Presburger/IntegerRelation.h) \
             — MLIR upstream, out of campaign scope"
        )
    }
}

/// `LiveRangeReductionPass::ExprInfo` (`LiveRangeReduction.cpp:116`) — ONE FLATTENED AFFINE
/// EXPRESSION: its coefficients, the SSA values they multiply, and the local-variable constraints
/// `getFlattenedAffineExpr` produced alongside them.
///
/// ⛔ NOT `PropagationAnalysis::ExprInfo` (`Analyses/PropagationAnalysis.h:38`) — a different class of
/// the same name, in the out-of-scope `Analyses/` directory, and the one
/// `crustify-senpass/OUTSIDE-DEPS.tsv` flags for this pass. This one is the pass's own nested class.
///
/// ⛔ NO DERIVED `PartialEq`: `operator==` (`:125`) compares the two constraint systems with
/// `isEqual`, which is CONTENT equality; [`FlatAffineValueConstraints`] is held by identity, so a
/// derived `==` would answer a different question.
#[derive(Debug, Clone, Default)]
pub struct ExprInfo {
    /// `expr_coeffs_` — one per dim then per local var, the constant already popped off by
    /// `getBaseExpr` (`:320-321`).
    pub expr_coeffs: Vec<i64>,
    /// `args_` — the SSA values those coefficients multiply.
    pub args: Vec<Val>,
    /// `local_vars_constraints`.
    pub local_vars_constraints: FlatAffineValueConstraints,
}

impl ExprInfo {
    /// Replaces: e056_set
    ///
    /// Overwrites all three fields — the one way `getBaseExpr` (`:342`, `:350`) fills an `ExprInfo`.
    ///
    /// ⛔ `clearAndCopyFrom` IS A REPLACEMENT, NOT A MERGE: it drops whatever system was there and
    /// takes a copy, which owned assignment performs exactly.
    pub fn set(&mut self, expr_coeffs: Vec<i64>, args: Vec<Val>, cst: FlatAffineValueConstraints) {
        self.expr_coeffs = expr_coeffs;
        self.args = args;
        self.local_vars_constraints = cst;
    }

    /// Replaces: e057_print
    ///
    /// The three-section trace `printSsaMap` (`e304`) writes per mapped value, each header indented
    /// two spaces.
    ///
    /// ⭐ EVERY COEFFICIENT IS FOLLOWED BY `", "`, INCLUDING THE LAST (`:144`), and "constrains" is
    /// the reference's own spelling (`:152`) — this is its output text, not prose.
    ///
    /// ⛔ `Value::print` PRINTS THE DEFINING OPERATION, not an SSA name, falling back to
    /// `<block argument> of type '..' at index: N`; [`print::emit`] already terminates the line that
    /// the reference's `os << "\n"` adds after `Operation::print`.
    pub fn print(&self, defs: Definitions<'_>, out: &mut String) {
        out.push_str("  Expression coeffs: ");
        for item in &self.expr_coeffs {
            let _ = write!(out, "{item}, ");
        }
        out.push_str("\n  SSA variables: \n");
        for arg in &self.args {
            match defs.of(*arg) {
                Some(op) => print::emit(out, op, 0),
                // Every `args_` element is an affine dim operand, so `index`-typed.
                None => match defs.for_arg_of(*arg) {
                    Some((_, index)) => {
                        let _ = writeln!(
                            out,
                            "<block argument> of type 'index' at index: {index}"
                        );
                    }
                    None => out.push('\n'),
                },
            }
        }
        out.push_str("  local variable constrains: \n");
        self.local_vars_constraints.write(out);
    }
}

/// Replaces: e058_areElementSizeIdentical
///
/// Whether two addresses step by the SAME, KNOWN element width — the guard `findDominantValue`
/// (`e441`) puts in front of commoning two values' live ranges.
///
/// ⛔ AN UNKNOWN WIDTH IS NOT A MATCH. The reference's `< 1` tests are what reject
/// [`element_size`]'s `-1`, so two values that BOTH have no `element_size` answer `false` rather
/// than "identical" — which is also why the four island gaps [`element_size`] documents are
/// conservative here rather than wrong.
#[must_use]
pub fn are_element_size_identical(lhs: Val, rhs: Val, defs: Definitions<'_>) -> bool {
    match (element_size(lhs, defs), element_size(rhs, defs)) {
        (Some(lhs_size), Some(rhs_size)) => {
            lhs_size.0 >= 1 && rhs_size.0 >= 1 && lhs_size == rhs_size
        }
        _ => false,
    }
}

/// Replaces: e059_getFirstGlobalOrConstantAncestor
///
/// Walks an address back through the ops that merely OFFSET it until it reaches the constant, the
/// per-core query or the symbol it is founded on, stopping at the first op outside `uniform_region`.
///
/// ⛔ `None` IS `signalPassFailure()`: an op the walk cannot attribute an address operand to makes
/// the reference `emitOpError("can't find global ancestor")` and fail the pass (`:1153-1154`).
///
/// ⛔ AND THE SRC/DST SPLIT IS THE REFERENCE'S INTENT, NOT ITS CODE. `val ==
/// load_store_op.getSrcMutableAddr()` (`:1127`) compares an op RESULT against that op's own operand
/// 2 and so is always false, leaving the `src` branch dead and every `load_and_store` address
/// walking through `dst`; the `.td` names the two results `src_res`/`dst_res` as "the final,
/// potentially updated, address" of each side (`SentientOps.td:744-746`), so this dispatches on WHICH
/// RESULT `val` is.
#[must_use]
pub fn get_first_global_or_constant_ancestor(
    val: Val,
    uniform_region: &[Op],
    defs: Definitions<'_>,
) -> Option<Val> {
    let mut val = val;
    loop {
        // ⭐ `DT_CHECK(isa<sentient::ForOp>(op))` (`:1114`) IS A FACT ABOUT THE ISLAND, not a check:
        // `for` is the only op of this dialect that binds a region argument.
        let (op, next) = match defs.for_arg_of(val) {
            Some((op @ Op::Sentient(sentient::Op::For { bound, carried, .. }), index)) => {
                // Operands are `[bound, initArgs..]` and arguments `[iv, iterArgs..]`, so
                // `getOperand(getArgNumber())` is the bound for the iv and `carried[i - 1].init`
                // for iter arg `i - 1`.
                let next = index
                    .checked_sub(1)
                    .map_or(Some(*bound), |position| {
                        carried.get(position).map(|value| value.init)
                    })?;
                (op, next)
            }
            _ => {
                let op = defs.of(val)?;
                let next = match op {
                    Op::Sentient(
                        sentient::Op::LoadAndSend { mutable_addr, .. }
                        | sentient::Op::ReceiveAndStore { mutable_addr, .. }
                        | sentient::Op::LoadAndExtractScalar { mutable_addr, .. }
                        | sentient::Op::LoadComputeAndSend { mutable_addr, .. },
                    ) => *mutable_addr,
                    Op::Sentient(sentient::Op::LoadAndStore {
                        src_mutable_addr,
                        dst_mutable_addr,
                        results,
                        ..
                    }) => {
                        if val == results.0 {
                            *src_mutable_addr
                        } else {
                            *dst_mutable_addr
                        }
                    }
                    // ⭐ FOLLOW THE OPERAND THAT IS NOT A CONSTANT, PREFERRING `lhs` (`:1141-1146`)
                    // — an address plus a constant offset is still that address.
                    Op::Sentient(
                        sentient::Op::ScalarAdd { lhs, rhs, .. }
                        | sentient::Op::ScalarSub { lhs, rhs, .. },
                    ) => {
                        if is_sentient_constant(*lhs, defs) {
                            *rhs
                        } else {
                            *lhs
                        }
                    }
                    // `getOperand(getResultNumber() + 1)` — result `i` came from `initArgs[i]`.
                    Op::Sentient(sentient::Op::For { carried, .. }) => {
                        carried
                            .iter()
                            .find(|value| value.result == val)
                            .map(|value| value.init)?
                    }
                    // ⭐ THE THREE ANCESTORS THE WALK IS LOOKING FOR — returned WITHOUT the region
                    // test, which is why a terminal op outside the region still answers (`:1151`).
                    Op::Sentient(sentient::Op::ScalarConstant { .. })
                    | Op::Uniform(uniform::Op::QueryMap { .. })
                    | Op::Symbol(symbol::Op::CreateSymbol { .. }) => return Some(val),
                    // `emitOpError("can't find global ancestor"); signalPassFailure();`
                    _ => return None,
                };
                (op, next)
            }
        };
        // `if (op->getParentRegion() != uniform_region) break;` — the IMMEDIATE parent, so
        // membership at the top level of the region's op list.
        if !uniform_region.contains(op) {
            return Some(val);
        }
        val = next;
    }
}

/// `dcc::utils::isConstant<sentient::ConstantOp>` (`Utils/Utils.cpp:423`) at the one instantiation
/// this pass uses — a `sentient.scalar_constant`, or a `uniform.query_map` every one of whose
/// per-core values is one.
fn is_sentient_constant(val: Val, defs: Definitions<'_>) -> bool {
    let is_constant_op = |op: Option<&Op>| {
        matches!(
            op,
            Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
        )
    };
    match defs.of(val) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => match defs.of(*map) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs
                .iter()
                .all(|(_, value)| is_constant_op(defs.of(*value))),
            _ => false,
        },
        op => is_constant_op(op),
    }
}

/// Replaces: e060_areReglocalesMatching
///
/// Whether two values live in the same register file — the second half of `findDominantValue`'s
/// (`e441`) commoning guard.
///
/// ⛔ TWO UNASSIGNED VALUES MATCH. [`sentient::RegType::Unknown`] is a locale like any other to an
/// `==`, and the reference compares the enums directly; before `RegisterTypeAssignment` (D66) has
/// run that is every value in the module.
#[must_use]
pub fn are_reglocales_matching(lhs: Val, rhs: Val, defs: Definitions<'_>) -> bool {
    value_reg_locale(lhs, defs) == value_reg_locale(rhs, defs)
}

/// ONE EQUIVALENCE CLASS OF THE PASS'S SSA MAP — `expr_info_list_[i]` and `ssa_value_list_[i]`
/// (`LiveRangeReduction.cpp:104-106`) as one record.
///
/// ⭐ ONE STRUCT WHERE THE REFERENCE KEEPS TWO PARALLEL VECTORS, because `addToMap` (`e440`, `:241`)
/// only ever grows the two together — so a length disagreement between them is unwritable here.
#[derive(Debug, Clone, Default)]
pub struct EquivalenceClass {
    /// `expr_info_list_[i]` — the flattened affine expression every value of this class shares.
    pub expr_info: ExprInfo,
    /// `ssa_value_list_[i]` — the values that share it, in discovery order.
    pub values: Vec<Val>,
}

/// `LiveRangeReductionPass`'S SSA MAP — the four members `mapAllValues` (`e503`) fills and
/// `reduceLiveRange` (`e560`) consumes (`:104-108`).
#[derive(Debug, Clone, Default)]
pub struct SsaMap {
    /// `expr_info_list_` zipped with `ssa_value_list_` — see [`EquivalenceClass`].
    pub classes: Vec<EquivalenceClass>,
    /// `ssa_expr_const_map_` — one constant offset per unit, for each mapped value.
    pub const_offsets: BTreeMap<Val, Vec<i64>>,
    /// `ssa_negated_map_` — whether the value's coefficients were sign-flipped by `getBaseExpr`.
    pub negated: BTreeMap<Val, bool>,
}

impl SsaMap {
    /// Replaces: e304_printSsaMap
    ///
    /// The `DEBUG_WITH_TYPE` dump of the whole map — one section per equivalence class, one block per
    /// value in it.
    ///
    /// ⛔ EVERY CLASS'S SECTION RUNS INTO [`ExprInfo::print`], whose tail is MLIR upstream's
    /// `FlatAffineValueConstraints::print` and out of campaign scope.
    pub fn print_ssa_map(&self, defs: Definitions<'_>, out: &mut String) {
        for class in &self.classes {
            out.push_str("+++printing ExprInfo:\n");
            class.expr_info.print(defs, out);
            for value in &class.values {
                out.push_str("--\n");
                self.print_mapped_value(*value, defs, out);
            }
            out.push_str("\n+++++++++++++\n\n");
        }
    }

    /// One mapped value's block of the dump — the value itself, its per-unit constant offsets, and
    /// whether its coefficients were negated (`:225-232`).
    ///
    /// ⛔ A VALUE WITH NO ENTRY PRINTS AS EMPTY AND `false`, NOT AS MISSING: both members are read
    /// through `DenseMap::operator[]`, which DEFAULT-CONSTRUCTS on a miss.
    /// ⭐ `os.indent(2) << "\nconst offset: "` PUTS THE TWO SPACES *BEFORE* THE NEWLINE (`:227`) —
    /// the reference's own output, kept character for character.
    fn print_mapped_value(&self, value: Val, defs: Definitions<'_>, out: &mut String) {
        print_value(value, defs, out);
        out.push_str("  \nconst offset: ");
        for item in self.const_offsets.get(&value).into_iter().flatten() {
            let _ = write!(out, "{item}, ");
        }
        let negated = self.negated.get(&value).copied().unwrap_or(false);
        let _ = write!(
            out,
            "\nexpr negated: {}\n",
            if negated { "true" } else { "false" }
        );
    }
}

/// `Value::print` — the DEFINING OPERATION, or the block-argument fallback.
///
/// ⛔ [`ExprInfo::print`] INLINES THE SAME THREE LINES rather than calling this: that anchor is
/// already filled, and this port does not reach into completed work to share a helper.
fn print_value(val: Val, defs: Definitions<'_>, out: &mut String) {
    match defs.of(val) {
        Some(op) => print::emit(out, op, 0),
        None => match defs.for_arg_of(val) {
            Some((_, index)) => {
                let _ = writeln!(out, "<block argument> of type 'index' at index: {index}");
            }
            None => out.push('\n'),
        },
    }
}

/// `AffineMap` AS THIS PASS USES ONE — an identity, plus the single structural fact `getBaseExpr`
/// reads off it.
///
/// ⛔ MLIR UPSTREAM AND OUT OF CAMPAIGN SCOPE, exactly like [`FlatAffineValueConstraints`]: the map is
/// built by `PropagationAnalysis` and flattened by `mlir::getFlattenedAffineExpr`, neither of which is
/// in this scope. `getResult(0)` is the only expression ever taken from it (`:308`), so the map does
/// not have to be indexable here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PropagatedMap {
    /// WHICH map — `propagated_map_`, held by identity.
    pub id: u32,
    /// `getNumDims()` (`:328`) — how many of the flattened coefficients are dimension coefficients.
    pub num_dims: usize,
}

/// `PropagationAnalysis::ExprInfo` (`Analyses/PropagationAnalysis.h:38`) — ONE UNIT'S PROPAGATED
/// EXPRESSION.
///
/// ⛔ NOT THIS PASS'S OWN [`ExprInfo`], which is a different class of the same name (`:116`).
#[derive(Debug, Clone, Default)]
pub struct PropagatedExpr {
    /// `propagated_map_`.
    pub propagated_map: PropagatedMap,
    /// `propagated_args_` — the SSA values the map's dimensions stand for.
    pub propagated_args: Vec<Val>,
    /// `cannot_be_resolved_`.
    pub cannot_be_resolved: bool,
}

/// `mlir::getFlattenedAffineExpr`'S THREE OUTPUTS for one expression (`:308-310`).
#[derive(Debug, Clone, Default)]
pub struct FlattenedExpr {
    /// `flat_expr` — one coefficient per dim, then per local var, then the constant LAST.
    pub coeffs: Vec<i64>,
    /// The local-variable system built alongside them.
    pub constraints: FlatAffineValueConstraints,
    /// `local_vars_constraints.getNumLocalVars()`.
    pub num_local_vars: usize,
}

/// `PropagationAnalysis::ExprInfoMap` (`Analyses/PropagationAnalysis.h:106`) — the BUCKETS of
/// propagated expressions for one value, and which bucket each unit reads.
///
/// ⭐ [`Self::buckets`]`.len()` IS `getUnitNumber()`, so the unit count and the unit→bucket map cannot
/// disagree; `unit_number() == 0` is the reference's `isGlobal()`.
#[derive(Debug, Clone, Default)]
pub struct ExprInfoMap {
    /// `getExprInfoList()` — `None` is the reference's null bucket.
    pub exprs: Vec<Option<PropagatedExpr>>,
    /// `getListIdxFromUnitIdx(i)` for every unit `i`, in unit order.
    pub buckets: Vec<usize>,
}

impl ExprInfoMap {
    /// `getUnitNumber()` (`:223`).
    #[must_use]
    pub fn unit_number(&self) -> usize {
        self.buckets.len()
    }
}

/// `PropagationAnalysis` (`Analyses/PropagationAnalysis.h`) — THE SEAM ONTO AN ANALYSIS THIS CAMPAIGN
/// DOES NOT PORT, the same arrangement [`crate::transform::sentient::analyses`] uses for the others.
///
/// ⛔ SCOPED TO THIS FILE UNTIL A SECOND CONSUMER APPEARS. Hoist it beside those seams when one does;
/// `crustify-senpass/OUTSIDE-DEPS.tsv` names this analysis for several more passes.
pub trait PropagationAnalysis {
    /// WHAT `getUnitIndexMap()` (`Analyses/PropagationAnalysis.h:325`) ANSWERS — an associated type
    /// rather than a `dyn`, because `mapAllValues` copies the map into a pass member (`:830`) and
    /// `createMapAndQuery` reads it back a whole arm later.
    type Units: UnitIndexMap;

    /// `getAffineExpression(Value)` (`Analyses/PropagationAnalysis.h:308`).
    fn affine_expression(&mut self, val: Val) -> ExprInfoMap;

    /// `mlir::getFlattenedAffineExpr(map.getResult(0), map.getNumDims(), 0, ..)` (`:308-310`).
    fn flattened_affine_expr(&self, map: PropagatedMap) -> FlattenedExpr;

    /// `expr_prop.getUnitIndexMap()` (`:830`).
    fn unit_index_map(&self) -> Self::Units;

    /// `isPropagationSuccessful()` (`Analyses/PropagationAnalysis.h`), read once by e598.
    ///
    /// ⛔ DEFAULTED TO A `todo!` AND NOT TO `true`: the analysis that decides it is out of campaign
    /// scope, and answering "yes" for it would reduce live ranges against expressions nothing
    /// propagated — a fabricated analysis result.
    fn is_propagation_successful(&self) -> bool {
        todo!(
            "PropagationAnalysis::isPropagationSuccessful \
             (dcc/src/Transform/Sentient/Analyses/PropagationAnalysis.h) — out of campaign scope"
        )
    }
}

/// THE ANALYSIS THIS CAMPAIGN DOES NOT PORT — every method `todo!`s, naming it.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopePropagationAnalysis;

impl PropagationAnalysis for OutOfScopePropagationAnalysis {
    type Units = OutOfScopeUnitIndexMap;

    fn affine_expression(&mut self, _val: Val) -> ExprInfoMap {
        todo!(
            "PropagationAnalysis::getAffineExpression \
             (dcc/src/Transform/Sentient/Analyses/PropagationAnalysis.h) — out of campaign scope"
        )
    }

    fn flattened_affine_expr(&self, _map: PropagatedMap) -> FlattenedExpr {
        todo!(
            "mlir::getFlattenedAffineExpr (mlir/Dialect/Affine/Analysis/AffineStructures.h) \
             — MLIR upstream, out of campaign scope"
        )
    }

    fn unit_index_map(&self) -> Self::Units {
        OutOfScopeUnitIndexMap
    }
}

/// WHAT `getBaseExpr` ANSWERS WITH — the three out-parameters it fills on `LogicalResult::success()`.
#[derive(Debug, Clone, Default)]
pub struct BaseExpr {
    /// `expr_info` — the one flattened expression every unit agreed on.
    pub expr_info: ExprInfo,
    /// `const_val` — ONE CONSTANT PER UNIT, `i64::MAX` where no expression applies.
    pub const_val: Vec<i64>,
    /// `negated` — whether the agreed coefficients were sign-flipped to make the first non-zero one
    /// positive.
    pub negated: bool,
}

/// Replaces: e305_getBaseExpr
///
/// One value's base expression: flatten every unit's propagated map, insist they all agree, and hand
/// back the per-unit constant offsets they differ by.
///
/// ⛔ `None` IS `LogicalResult::failure()` — no unit resolved, or two of them disagreed.
/// ⛔ `i64::MAX` IS THE "NO EXPRESSION FOR THIS UNIT" SENTINEL, written by three separate paths, then
/// smoothed away for a `uniform.uniformize_regions` result because those live in the global region.
/// ⭐ THE `args.size() > dim_num` TRIM IS THE REFERENCE'S OWN WORKAROUND for a bug in expression
/// propagation (`:333-336`) — kept, comment and all.
#[must_use]
pub fn get_base_expr(
    analysis: &mut impl PropagationAnalysis,
    value: Val,
    defs: Definitions<'_>,
) -> Option<BaseExpr> {
    let expr_info_map = analysis.affine_expression(value);
    let is_constant = is_sentient_constant(value, defs);
    // `isa<OpResult>(value)`, whose complement is `isa<BlockArgument>(value)`.
    let is_op_result = defs.of(value).is_some();
    let mut expr_info = ExprInfo::default();
    let mut negated = false;
    let mut is_expr_info_known = false;
    let mut num_irresolvable = 0usize;
    let mut const_val_tmp: Vec<i64> = Vec::new();
    // Each bucket here may correspond to one or multiple units.
    for expr in &expr_info_map.exprs {
        // if no expression, use INT64_MAX to indicate no applicable expression for this unit
        let Some(expr) = expr else {
            const_val_tmp.push(i64::MAX);
            continue;
        };
        // for constant values, const_val needs to be filled.
        if expr.cannot_be_resolved && !is_constant {
            num_irresolvable += 1;
            // if expressions for all cores are irresolvable, return failure
            // if value is blockArgument, use itself as baseExpr
            if num_irresolvable == expr_info_map.exprs.len() && is_op_result {
                return None;
            }
            const_val_tmp.push(i64::MAX);
            continue;
        }

        let flat = analysis.flattened_affine_expr(expr.propagated_map);
        let mut coeffs = flat.coeffs;
        let dim_num = expr.propagated_map.num_dims;
        let mut args = expr.propagated_args.clone();
        // TODO there is a bug in expression propagation pass which incorrectly adds
        // additional value to args_. a temporary fix to remove the first entry.
        if args.len() > dim_num {
            args.remove(0);
        }
        // ⛔ `DT_CHECK(flat_expr.size() == dim_num + getNumLocalVars() + 1)` (`:337-338`).
        if coeffs.len() != dim_num + flat.num_local_vars + 1 {
            panic!(
                "a flattened affine expression has {} coefficients, not one per dimension, one per \
                 local variable and one constant (LiveRangeReduction.cpp:337-338)",
                coeffs.len()
            );
        }

        // the const expr is at the back.
        const_val_tmp.push(coeffs.pop().unwrap_or(i64::MAX));
        // normalize flat_expr to make the first non-zero value positive.
        let temp_negated = coeffs
            .iter()
            .find(|coeff| **coeff != 0)
            .is_some_and(|coeff| *coeff < 0);
        if temp_negated {
            for coeff in &mut coeffs {
                // ⭐ `-x` ON `i64::MIN` IS THE REFERENCE'S OWN TWO'S-COMPLEMENT WRAP.
                *coeff = coeff.wrapping_neg();
            }
        }
        if is_expr_info_known {
            // compare the new expr with the existing one
            let mut new_expr_info = ExprInfo::default();
            new_expr_info.set(coeffs, args, flat.constraints);
            // If mismatching expr_info (e.g. due to different coefficients), return failure.
            if !expr_info_eq(&new_expr_info, &expr_info) || negated != temp_negated {
                return None;
            }
        } else {
            // set expr_info for the first time
            expr_info.set(coeffs, args, flat.constraints);
            negated = temp_negated;
            is_expr_info_known = true;
        }
    }

    // At this point const_val_tmp contains one constant per bucket. However, we need one per unit.
    // ⭐ A BUCKET INDEX PAST THE LIST TAKES THE SENTINEL — the same "no expression for this unit"
    // answer the null bucket gives, and only reachable if the analysis disagreed with itself.
    let mut const_val: Vec<i64> = expr_info_map
        .buckets
        .iter()
        .map(|bucket| const_val_tmp.get(*bucket).copied().unwrap_or(i64::MAX))
        .collect();

    // if iter_arg is not resolvable for all units, copy the last entry of const_val to all others
    if num_irresolvable == expr_info_map.exprs.len()
        && expr_info_map.exprs.len() > 1
        && !is_op_result
    {
        if let Some(last) = const_val.last().copied() {
            // ⭐ SATURATING WHERE `const_val.size() - 1` WOULD WRAP on an empty vector (`:388`).
            let count = const_val.len().saturating_sub(1);
            for item in const_val.iter_mut().take(count) {
                *item = last;
            }
        }
    }

    // uniformize_regionsOp's results are in global region, so it is necessary to replace INT64_MAX
    // entries with any valid value in the vector to improve the uniformity across cores.
    if is_op_result && defs.of(value).is_some_and(is_uniformize_regions) {
        let mut tmp = i64::MAX;
        for item in &const_val {
            if *item != i64::MAX {
                tmp = *item;
            }
        }
        for item in &mut const_val {
            if *item == i64::MAX {
                *item = tmp;
            }
        }
    }

    Some(BaseExpr {
        expr_info,
        const_val,
        negated,
    })
}

/// `ExprInfo::operator==` (`:125`) — coefficients, then args, then the two constraint SYSTEMS.
///
/// ⛔ NOT A DERIVED `PartialEq`, AND THE ORDER IS THE REFERENCE'S: the element-wise tests answer
/// first, so the Presburger `isEqual` that [`FlatAffineValueConstraints`] cannot express is reached
/// only when every coefficient and every arg already matched.
/// ⭐ TWO IDENTICAL SYSTEMS ANSWER `true` WITHOUT IT — and that is the only case `getBaseExpr`
/// produces, since one `getFlattenedAffineExpr` call feeds both sides of the comparison.
fn expr_info_eq(lhs: &ExprInfo, rhs: &ExprInfo) -> bool {
    if lhs.expr_coeffs != rhs.expr_coeffs || lhs.args != rhs.args {
        return false;
    }
    if lhs.local_vars_constraints.0 == rhs.local_vars_constraints.0 {
        return true;
    }
    todo!(
        "affine::FlatAffineValueConstraints::isEqual (mlir/Analysis/Presburger/IntegerRelation.h) \
         — MLIR upstream, out of campaign scope"
    )
}

/// Replaces: e306_getRootIterArg
///
/// Walks an address back through every op that merely FORWARDS it — adds, subs, loop carries,
/// transfers, local regions — to the value it is rooted in.
///
/// ⛔ ONLY THE ADDRESS RESULT OF A `load_and_extract_scalar` CHAINS: the data result represents what
/// is STORED at that address and maps to nothing in the IR, so it ends the walk (`:456-461`).
/// ⭐ IT ALWAYS ANSWERS WITH A VALUE, NEVER `None`: an op with no forwarding arm ends the walk at the
/// result it was asked about, which is the reference's `break`.
#[must_use]
pub fn get_root_iter_arg(val: Val, defs: Definitions<'_>) -> Val {
    let mut val = val;
    // `while (val && !isa<BlockArgument>(val))` — a block argument has no defining op.
    while let Some(op) = defs.of(val) {
        let next = match op {
            // Follow the operand that is not a constant, preferring `$inp1` (`:422-434`).
            Op::Sentient(
                sentient::Op::ScalarAdd { lhs, rhs, .. } | sentient::Op::ScalarSub { lhs, rhs, .. },
            ) => {
                if defs.of(*lhs).is_none() || !is_sentient_constant(*lhs, defs) {
                    *lhs
                } else {
                    *rhs
                }
            }
            // `getOperand(getResultNumber() + getNumControlOperands())` — the one control operand is
            // `$bound`, so result `i` came from `initArgs[i]`.
            Op::Sentient(sentient::Op::For { carried, .. }) => {
                match carried.iter().find(|value| value.result == val) {
                    Some(value) => value.init,
                    None => break,
                }
            }
            Op::Sentient(
                sentient::Op::LoadAndSend { mutable_addr, .. }
                | sentient::Op::ReceiveAndStore { mutable_addr, .. }
                | sentient::Op::LoadComputeAndSend { mutable_addr, .. },
            ) => *mutable_addr,
            // `result_idx > 0 ? getDstMutableAddr() : getSrcMutableAddr()` — `results.0` is `$src_res`.
            Op::Sentient(sentient::Op::LoadAndStore {
                src_mutable_addr,
                dst_mutable_addr,
                results,
                ..
            }) => {
                if val == results.0 {
                    *src_mutable_addr
                } else {
                    *dst_mutable_addr
                }
            }
            Op::Sentient(sentient::Op::LoadAndExtractScalar {
                mutable_addr,
                addr_result,
                ..
            }) => {
                if val == *addr_result {
                    *mutable_addr
                } else {
                    break;
                }
            }
            local if is_uniformize_regions(local) => match uniformize_result_source(local, val) {
                Some(source) => source,
                None => break,
            },
            _ => break,
        };
        val = next;
    }
    val
}

/// `dyn_cast<uniform::UniformizeRegionsOp>(op)` IN BOTH ISLAND SPELLINGS — the same op whether its
/// regions hold this rung's ops or the rung below's, which is how `SimplifyUniformRegions` reads it.
fn is_uniformize_regions(op: &Op) -> bool {
    matches!(
        op,
        Op::Uniform(uniform::Op::UniformizeRegions { .. })
            | Op::UniformRegions(UniformRegions::UniformizeRegions { .. })
    )
}

/// `uniformize_op->getRegion(0).front().getTerminator()->getOperand(result_idx)` — what region 0
/// yields for the result `val` (`:467-478`), in both island spellings.
///
/// ⛔ THE TWO `DT_CHECK`s ARE A SHAPE, NOT A CHOICE: more than one region means EXACTLY two, the
/// second holding nothing but its terminator (`:470-476`).
fn uniformize_result_source(op: &Op, val: Val) -> Option<Val> {
    let (results, yielded, region_count, second_len) = match op {
        Op::UniformRegions(UniformRegions::UniformizeRegions {
            regions, results, ..
        }) => (
            results.as_slice(),
            regions.first().and_then(|region| match region.body.last() {
                Some(Op::Uniform(uniform::Op::Yield { operands })) => Some(operands.as_slice()),
                _ => None,
            }),
            regions.len(),
            regions.get(1).map(|region| region.body.len()),
        ),
        Op::Uniform(uniform::Op::UniformizeRegions { regions, results }) => (
            results.as_slice(),
            regions.first().and_then(|region| match region.body.last() {
                Some(lower::Op::Uniform(uniform::Op::Yield { operands })) => {
                    Some(operands.as_slice())
                }
                _ => None,
            }),
            regions.len(),
            regions.get(1).map(|region| region.body.len()),
        ),
        _ => return None,
    };
    if region_count > 1 {
        if region_count != 2 {
            panic!(
                "a uniform.uniformize_regions has {region_count} regions, not one or two \
                 (LiveRangeReduction.cpp:470)"
            );
        }
        if second_len != Some(1) {
            panic!(
                "a uniform.uniformize_regions' second region holds more than its terminator \
                 (LiveRangeReduction.cpp:471-476)"
            );
        }
    }
    let at = results.iter().position(|result| *result == val)?;
    yielded?.get(at).copied()
}

/// THE SECOND REGION OF A `uniform.uniformize_regions`, WITH THE REGIONS BEFORE IT — what
/// `cloneValueToRegion` is handed.
///
/// ⛔⛔ `DT_CHECK(region.getRegionNumber() == 1)` (`:920`) IS THIS TYPE, not a check: only
/// [`Self::of`] can build one, and it answers `None` for an op that has no second region.
/// ⭐ [`Self::preceding`] IS THE OTHER HALF OF THE REFERENCE'S `isProperAncestor` TEST: a value
/// defined in an ENCLOSING region needs no clone, and a value defined in a SIBLING region is exactly
/// one found in here.
pub struct SecondRegion<'a> {
    /// Region 1 — where the clone goes.
    pub region: &'a mut LocalRegion,
    /// Regions `0..1` — where a sibling region's definition is found.
    pub preceding: &'a [LocalRegion],
}

impl<'a> SecondRegion<'a> {
    /// Region 1 of `op`, or `None` when it has none.
    pub fn of(op: &'a mut UniformRegions) -> Option<SecondRegion<'a>> {
        let regions = op.regions_mut();
        if regions.len() < 2 {
            return None;
        }
        let (preceding, rest) = regions.split_at_mut(1);
        Some(SecondRegion {
            region: rest.first_mut()?,
            preceding,
        })
    }
}

/// Replaces: e307_cloneValueToRegion
///
/// Makes a value defined in region 0 usable from region 1 by re-minting, INSIDE region 1, the
/// `uniform.query_map` behind it over a mapping of region 1's own units.
///
/// ⛔ `None` MEANS THE MAPPING DOES NOT COVER REGION 1'S UNITS, and the caller (`e442`) then erases the
/// op it was building and keeps the original (`LiveRangeReduction.cpp:1083-1087`).
/// ⛔ THE `!query_op->isProperAncestor(region.getParentOp())` GUARD IS DEAD CODE (`:929`): a
/// `uniform.query_map` carries no regions, so it can never be an ancestor of anything and the mapping
/// is ALWAYS re-minted. Ported as the reference behaves, not as it reads.
pub fn clone_value_to_region(val: Val, target: SecondRegion<'_>, vals: &mut Values) -> Option<Val> {
    let SecondRegion { region, preceding } = target;
    // `isa<OpResult>(val) && !getParentRegion()->isProperAncestor(&region)` — a sibling region's op.
    let bodies: Vec<&[Op]> = preceding
        .iter()
        .map(|earlier| earlier.body.as_slice())
        .collect();
    let defs = Definitions::from_innermost(&bodies);
    let Some(op) = defs.of(val) else {
        return Some(val);
    };
    // ⛔ `DT_CHECK(isConstant<sentient::ConstantOp>(val))` (`:923`).
    if !is_sentient_constant(val, defs) {
        panic!(
            "a non-constant value defined in a sibling local region cannot be cloned into region 1 \
             (LiveRangeReduction.cpp:923)"
        );
    }
    let Op::Uniform(uniform::Op::QueryMap { map, .. }) = op else {
        return Some(val);
    };
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(*map) else {
        panic!(
            "a uniform.query_map's $map is not defined by a uniform.def_immutable_mapping \
             (LiveRangeReduction.cpp:925-926)"
        );
    };
    // check if immutable map contains values for the region's units — `collectUnitVals(
    // getRegionUnitList(1))`, then `getNonNullValuesFromKeys`, whose MISSES ARE DROPPED.
    let keys = region.units.clone();
    let values: Vec<Val> = keys
        .iter()
        .filter_map(|sought| {
            pairs
                .iter()
                .find(|(mapped, _)| mapped == sought)
                .map(|(_, value)| *value)
        })
        .collect();
    if values.len() != keys.len() {
        // if val is a queryOp which doesn't cover region's units, quit adding
        return None;
    }
    let mapping = vals.mint();
    let result = vals.mint();
    // ⭐ THE BUILDER'S INSERTION POINT IS INSIDE THE REGION, so before its `uniform.yield`.
    let at = region.body.len().saturating_sub(1);
    region.body.insert(
        at,
        Op::Uniform(uniform::Op::DefImmutableMapping {
            result: mapping,
            pairs: keys.into_iter().zip(values).collect(),
        }),
    );
    region.body.insert(
        at + 1,
        Op::Uniform(uniform::Op::QueryMap {
            result,
            map: mapping,
            key: region.arg,
        }),
    );
    Some(result)
}

impl SsaMap {
    /// Replaces: e440_addToMap
    ///
    /// Files one value under the equivalence class of its base expression, recording the per-unit
    /// constant offsets and the negation it differs from that class by.
    ///
    /// ⛔ A CONSTANT IS RECORDED BUT NEVER CLASSED (`:252-253`, `:259-261`): its live range needs no
    /// reducing, so it neither joins a class nor opens one — yet its offsets and its negation ARE
    /// mapped, because that constant term is what `reconstructOperation` substitutes with.
    /// ⛔ A VALUE WHOSE BASE EXPRESSION FAILS IS NOT MAPPED AT ALL, not mapped as empty.
    pub fn add_to_map(
        &mut self,
        analysis: &mut impl PropagationAnalysis,
        value: Val,
        defs: Definitions<'_>,
    ) {
        let is_constant = is_sentient_constant(value, defs);
        let Some(base) = get_base_expr(analysis, value, defs) else {
            return;
        };
        let existing = self
            .classes
            .iter()
            .position(|class| expr_info_eq(&class.expr_info, &base.expr_info));
        if !is_constant {
            match existing {
                Some(index) => self.classes[index].values.push(value),
                None => self.classes.push(EquivalenceClass {
                    expr_info: base.expr_info,
                    values: vec![value],
                }),
            }
        }
        self.const_offsets.insert(value, base.const_val);
        self.negated.insert(value, base.negated);
    }
}

/// `skip_elem_size_` (`LiveRangeReduction.cpp:1287-1290`) — whether the unit's own type waives
/// [`are_element_size_identical`], which it does for a PT, PE or SFP unit and no other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementSizeCheck {
    /// `skip_elem_size_ == false` — the widths must match.
    Required,
    /// `skip_elem_size_ == true`.
    Skipped,
}

/// Replaces: e441_findDominantValue
///
/// Scans this class's earlier values backwards for the CLOSEST one that properly dominates the
/// current value, steps by the same element width and shares its register file; `None` is
/// `LogicalResult::failure()`.
///
/// ⛔ A `scalar_add`/`scalar_sub` MUST ALSO SHARE THE DEF-USE ROOT (`:505-512`) — two adds rooted in
/// different iter args cannot have their registers merged — and such a candidate is SKIPPED, the scan
/// continuing to the next rather than failing.
#[must_use]
pub fn find_dominant_value(
    ssa_list: &[Val],
    current_value_index: usize,
    check: ElementSizeCheck,
    unit_body: &[Op],
    defs: Definitions<'_>,
) -> Option<usize> {
    let current = ssa_list[current_value_index];
    // `ssa_list[current_value_index].getDefiningOp()`, which `:499` and `:505` then dereference
    // unchecked — so a block argument here is the reference's own null dereference.
    let Some(current_at) = utils::path_of(unit_body, current) else {
        panic!(
            "findDominantValue was asked about a value with no defining op in this program unit, \
             where the reference dereferences a null Operation* \
             (LiveRangeReduction.cpp:484,499,505)"
        )
    };
    let current_op = current_at.op(unit_body);
    for scanner in (0..current_value_index).rev() {
        let candidate = ssa_list[scanner];
        if !properly_dominates(candidate, &current_at, unit_body, defs) {
            continue;
        }
        if check == ElementSizeCheck::Required
            && !are_element_size_identical(candidate, current, defs)
        {
            continue;
        }
        if !are_reglocales_matching(candidate, current, defs) {
            continue;
        }
        if matches!(
            current_op,
            Some(Op::Sentient(
                sentient::Op::ScalarAdd { .. } | sentient::Op::ScalarSub { .. }
            ))
        ) && get_root_iter_arg(current, defs) != get_root_iter_arg(candidate, defs)
        {
            continue;
        }
        return Some(scanner);
    }
    None
}

/// `DominanceInfo::properlyDominates(Value, Operation*)` — the value's DEFINING OP against `use_at`
/// when it has one, and its BLOCK otherwise.
///
/// ⭐ THREE CASES, ALL THE ONE MLIR RULE. A result defined in this unit compares paths (and never
/// properly dominates its own op); a result defined outside it was bound in the enclosing program
/// scope and so dominates the whole unit; a block argument is a `sentient.for`'s `iv` or iter arg and
/// dominates exactly what that loop's body holds.
fn properly_dominates(
    val: Val,
    use_at: &OpAt,
    unit_body: &[Op],
    defs: Definitions<'_>,
) -> bool {
    if defs.of(val).is_some() {
        return match utils::path_of(unit_body, val) {
            Some(def_at) => def_at != *use_at && def_at.dominates(use_at),
            None => true,
        };
    }
    let mut at = use_at.clone();
    while let Some(parent) = at.parent() {
        if let Some(Op::Sentient(sentient::Op::For { iv, carried, .. })) = parent.op(unit_body)
            && (*iv == val || carried.iter().any(|value| value.arg == val))
        {
            return true;
        }
        at = parent;
    }
    false
}

/// WHAT `getQueryKeyAndUnitsFromParentRegion` (`e392`, `Transform/Sentient/Utils.cpp:40`) ANSWERS —
/// the enclosing region's own key and every unit it covers.
///
/// ⛔ A PARAMETER BECAUSE `e392` IS NOT PORTED YET, and it is the same seam [`SecondRegion`] is: the
/// climb out of the op to its parent `dataflow.program_unit` or sibling `uniform` region is mechanism
/// its owner writes once, while [`add_result_to_yield`] and [`create_map_and_query`] need only its
/// answer. ⭐ e442 READS `units` ALONE — the reference's own `key` is `(void)`-discarded there
/// (`LiveRangeReduction.cpp:986`) and only `e443` uses it.
#[derive(Debug, Clone)]
pub struct ParentRegionQuery {
    /// `key` — the parent region's argument, which a new `uniform.query_map` is keyed by.
    pub key: Val,
    /// `all_units` — every unit of the parent region's list.
    pub units: Vec<Val>,
}

/// Replaces: e442_addResultToYield
///
/// Rebuilds the `uniform.uniformize_regions` with ONE MORE RESULT: the op at `op_num` of region
/// `region_num` now yields its result `result_idx`, region 1 yields that value's global ancestor
/// re-minted for its own units, and the new result's element width and register file are appended.
///
/// ⛔ A ONE-REGION OP GAINS ITS SECOND REGION HERE, running on whatever of the parent's unit list
/// region 0 does not cover (`:981-997`) — the reference's second `listSizes` entry, which this
/// island spells as [`LocalRegion::units`].
/// ⛔ `None` FROM [`clone_value_to_region`] ABANDONS THE WHOLE REWRITE and the ORIGINAL op is
/// answered with (`:1083-1087`); `newly_added`/`ops_to_be_delected` (`:1093-1096`) are the rewriter's
/// memory management and become the caller storing this answer into the slot.
#[must_use]
pub fn add_result_to_yield(
    uniformize_op: &UniformRegions,
    parent: &ParentRegionQuery,
    region_num: usize,
    op_num: utils::InBlock,
    result_idx: usize,
    enclosing: &[&[Op]],
    vals: &mut Values,
) -> UniformRegions {
    let defs = Definitions::from_innermost(enclosing);
    // `uniform::UniformizeRegionsOp` IS THE PARAMETER'S TYPE (`:961-965`), and `$results` is what
    // only that variant carries.
    let UniformRegions::UniformizeRegions {
        regions,
        results,
        yielded,
    } = uniformize_op
    else {
        panic!(
            "addResultToYield takes a uniform.uniformize_regions, not a uniform.equalize_pattern \
             (LiveRangeReduction.cpp:961-965)"
        )
    };

    // `register_op.getResult(result_idx)` READ FROM THE ORIGINAL (`:1039`, `:1058-1067`): the clone
    // is a copy, so the width and the locale are the same on either side.
    let Some(source_result) = utils::operation_of_block(&regions[region_num].body, op_num)
        .and_then(|op| dialects::results(op).get(result_idx).copied())
    else {
        panic!(
            "addResultToYield was asked for result {result_idx} of op {} of a local region that has \
             neither, where the reference walks past the end of the block \
             (LiveRangeReduction.cpp:1039)",
            op_num.0
        )
    };

    // ⭐ THE SECOND REGION'S UNITS — `collectUnitVals(units)` against the parent's whole list
    // (`:981-997`). A two-region op keeps both lists as they are (`:1000-1006`).
    let unit_lists: Vec<Vec<Val>> = if regions.len() == 1 {
        let organized = dialects::collect_unit_ops(&regions[0].units, defs);
        vec![
            regions[0].units.clone(),
            parent
                .units
                .iter()
                .copied()
                .filter(|unit| !organized.contains(unit))
                .collect(),
        ]
    } else {
        regions.iter().map(|region| region.units.clone()).collect()
    };

    // THE TWO REGIONS OF THE NEW OP (`:1008-1026`), each binding a fresh argument.
    let mut new_regions: Vec<LocalRegion> = Vec::new();
    let mut cloned_result = source_result;
    for r in 0..2 {
        let arg = vals.mint();
        let body = match regions.get(r) {
            Some(source) if r == region_num || regions.len() == 2 => {
                let mut mapping = ValueMapping::new();
                mapping.map(source.arg, arg);
                let body = dialects::clone_ops(&source.body, vals, &mut mapping);
                if r == region_num {
                    cloned_result = mapping.lookup_or_default(source_result);
                }
                body
            }
            // A one-region op's new region 1 starts as a bare `uniform.yield` (`:1027-1030`).
            _ => vec![Op::Uniform(uniform::Op::Yield {
                operands: Vec::new(),
            })],
        };
        new_regions.push(LocalRegion {
            arg,
            units: unit_lists[r].clone(),
            body,
        });
    }

    let mut new_op = UniformRegions::UniformizeRegions {
        results: results
            .iter()
            .copied()
            .chain(core::iter::once(vals.mint()))
            .collect(),
        // ⭐ ONE ENTRY APPENDED TO WHATEVER WAS THERE (`:1043-1067`) — so an op that carried NEITHER
        // attribute over two results comes out with one entry for three, exactly as the reference's
        // `hasAttr`-guarded reads leave it.
        yielded: yielded
            .iter()
            .copied()
            .chain(core::iter::once(YieldedReg {
                element_size: element_size(source_result, defs),
                locale: value_reg_locale(source_result, defs),
            }))
            .collect(),
        regions: new_regions,
    };

    // `yield_args.push_back(register_op.getResult(result_idx))` (`:1039`).
    push_yield_operand(&mut new_op.regions_mut()[region_num], cloned_result);

    // THE OTHER REGION YIELDS THE ANCESTOR, RE-MINTED (`:1072-1090`). ⭐ `DT_CHECK(r == 1)` (`:1073`)
    // holds because the one caller only ever passes `region_num == 0` (`:1173-1174`).
    let scopes: Vec<&[Op]> = core::iter::once(new_op.regions()[0].body.as_slice())
        .chain(enclosing.iter().copied())
        .collect();
    let ancestor = get_first_global_or_constant_ancestor(
        cloned_result,
        &new_op.regions()[0].body,
        Definitions::from_innermost(&scopes),
    );
    let Some(ancestor) = ancestor else {
        panic!(
            "addResultToYield could not attribute the new result to a global ancestor, which is \
             `emitOpError(\"can't find global ancestor\")` and fails the pass \
             (LiveRangeReduction.cpp:1153-1154)"
        )
    };
    drop(scopes);
    let Some(target) = SecondRegion::of(&mut new_op) else {
        panic!(
            "the op addResultToYield just built has two regions \
             (LiveRangeReduction.cpp:1010-1011)"
        )
    };
    let Some(yield_operand) = clone_value_to_region(ancestor, target, vals) else {
        // if ancestor_val is a queryOp which doesn't cover region1's units, quit adding
        return uniformize_op.clone();
    };
    push_yield_operand(&mut new_op.regions_mut()[1], yield_operand);
    new_op
}

/// `yield_op->setOperands(yield_args)` WITH ONE MORE (`:1040`, `:1089`, `:1091`) — the terminator is
/// the last op of the region, and a region built by [`add_result_to_yield`] always has one.
fn push_yield_operand(region: &mut LocalRegion, operand: Val) {
    if let Some(Op::Uniform(uniform::Op::Yield { operands })) = region.body.last_mut() {
        operands.push(operand);
    }
}

/// WHAT [`create_map_and_query`] BUILT — the ops in the order the builder wrote them, and the value
/// the caller substitutes with.
#[derive(Debug, Clone)]
pub struct MapAndQuery {
    /// The `sentient.scalar_constant`s, then the `uniform.def_immutable_mapping`, then the
    /// `uniform.query_map` — what the reference leaves at the insertion point.
    pub ops: Vec<Op>,
    /// `query_op->getResult(0)`.
    pub result: Val,
}

/// Replaces: e443_createMapAndQuery
///
/// Turns one per-unit constant offset per unit into a `uniform.query_map` over a mapping from each of
/// the parent region's units to its own `sentient.scalar_constant`.
///
/// ⛔ ONLY THE UNITS THE PARENT REGION COVERS GET A KEY, AND IN THE REFERENCE'S REVERSED, GROUP-FLAT
/// ORDER — [`utils::select_indices_for_units`], whose own note explains the LIFO.
/// ⛔ A UNIT WHOSE OFFSET WAS THE `i64::MAX` SENTINEL HAS NO CONSTANT and the reference dereferences
/// the null it pushed for it (`:1222`, `:1249-1250`); that is the one `panic!` here.
#[must_use]
pub fn create_map_and_query(
    new_const_val: &[i64],
    ty: ScalarTy,
    parent: &ParentRegionQuery,
    index_map: &impl UnitIndexMap,
    defs: Definitions<'_>,
    vals: &mut Values,
) -> MapAndQuery {
    // create const_ops for all new constants — `INT64_MAX` pushes a NULL (`:1220-1229`).
    let constants: Vec<Option<(Val, Op)>> = new_const_val
        .iter()
        .map(|value| {
            (*value != i64::MAX).then(|| {
                let result = vals.mint();
                (
                    result,
                    Op::Sentient(sentient::Op::ScalarConstant {
                        value: *value,
                        result,
                        reg_locale: sentient::RegType::Imm,
                        ty,
                        is_symbol: false,
                    }),
                )
            })
        })
        .collect();

    // second, create an immutable map mapping getUnitOp to constOp — the reference's own LIFO walk.
    let mut units = parent.units.clone();
    let mut indices: Vec<UnitIndex> = Vec::new();
    utils::select_indices_for_units(&mut units, &mut indices, index_map, defs);
    let pairs: Vec<(Val, Val)> = units
        .iter()
        .zip(&indices)
        .map(|(unit, index)| {
            let Some(Some((result, _))) = constants.get(index.0 as usize) else {
                panic!(
                    "unit index {} has no constant of its own among {} — `const_ops.at(idx)` either \
                     throws or dereferences the null pushed for an INT64_MAX offset \
                     (LiveRangeReduction.cpp:1222, 1250-1252)",
                    index.0,
                    new_const_val.len()
                )
            };
            (*unit, *result)
        })
        .collect();

    let mapping = vals.mint();
    let result = vals.mint();
    let mut ops: Vec<Op> = constants
        .into_iter()
        .flatten()
        .map(|(_, op)| op)
        .collect();
    ops.push(Op::Uniform(uniform::Op::DefImmutableMapping {
        result: mapping,
        pairs,
    }));
    ops.push(Op::Uniform(uniform::Op::QueryMap {
        result,
        map: mapping,
        key: parent.key,
    }));
    MapAndQuery { ops, result }
}

/// WHAT `reconstructOperation` ANSWERS — a `LogicalResult` whose only `failure()` is the closing
/// `else`'s `op->emitError("unsupported op for live range reduction!")` (`:801-803`).
///
/// ⭐ NOT A `Result`: every profitability test in the body answers `success()` too, so "rewritten" and
/// "deliberately left alone" are ONE answer there and must stay one here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconstruction {
    /// `LogicalResult::success()` — rewritten, or left exactly as it was.
    Done,
    /// `LogicalResult::failure()`, which `e560_reduceLiveRange` turns into `signalPassFailure()`.
    Unsupported,
}

/// WHICH `$mutable_addr` OPERAND A TRANSFER OP'S RESULT IS A FUNCTION OF — `operand_idx` (`:670-704`),
/// named by the field rather than by the position the reference asks the op for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MutableAddr {
    /// `getMutableAddrMutable()` — the four one-address ops.
    Only,
    /// `getSrcMutableAddrMutable()`, which a `sentient.load_and_store`'s result 0 travels through.
    Src,
    /// `getDstMutableAddrMutable()`, which its result 1 does.
    Dst,
}

impl MutableAddr {
    /// `op->getOperand(operand_idx)`.
    fn read(self, op: &Op) -> Option<Val> {
        match (self, op) {
            (
                MutableAddr::Only,
                Op::Sentient(
                    sentient::Op::ReceiveAndStore { mutable_addr, .. }
                    | sentient::Op::LoadAndSend { mutable_addr, .. }
                    | sentient::Op::LoadAndExtractScalar { mutable_addr, .. }
                    | sentient::Op::LoadComputeAndSend { mutable_addr, .. },
                ),
            ) => Some(*mutable_addr),
            (
                MutableAddr::Src,
                Op::Sentient(sentient::Op::LoadAndStore {
                    src_mutable_addr, ..
                }),
            ) => Some(*src_mutable_addr),
            (
                MutableAddr::Dst,
                Op::Sentient(sentient::Op::LoadAndStore {
                    dst_mutable_addr, ..
                }),
            ) => Some(*dst_mutable_addr),
            _ => None,
        }
    }

    /// `op->setOperand(operand_idx, dominant_value)` (`:744`).
    fn write(self, op: &mut Op, val: Val) {
        match (self, op) {
            (
                MutableAddr::Only,
                Op::Sentient(
                    sentient::Op::ReceiveAndStore { mutable_addr, .. }
                    | sentient::Op::LoadAndSend { mutable_addr, .. }
                    | sentient::Op::LoadAndExtractScalar { mutable_addr, .. }
                    | sentient::Op::LoadComputeAndSend { mutable_addr, .. },
                ),
            ) => *mutable_addr = val,
            (
                MutableAddr::Src,
                Op::Sentient(sentient::Op::LoadAndStore {
                    src_mutable_addr, ..
                }),
            ) => *src_mutable_addr = val,
            (
                MutableAddr::Dst,
                Op::Sentient(sentient::Op::LoadAndStore {
                    dst_mutable_addr, ..
                }),
            ) => *dst_mutable_addr = val,
            _ => {}
        }
    }
}

/// WHICH OF `reconstructOperation`'S FIVE ARMS ONE VALUE'S DEFINING OP TAKES (`:561-805`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// `isa<AddOp, SubOp>` — THE ONLY ARM THAT BUILDS.
    AddOrSub,
    /// The five transfer ops. `None` is a `load_and_extract_scalar`'s DATA result, which is what is
    /// stored AT the address rather than a function of it, and answers `success()` (`:687-700`).
    Transfer(Option<MutableAddr>),
    /// `sentient.for`, and the `iter_args` position `resultNumber + getNumControlOperands()` names.
    Loop(usize),
    /// `isa<uniform::UniformizeRegionsOp>` — nothing to reconstruct (`:797-798`).
    Uniformize,
    /// The closing `else` (`:800-804`).
    Unsupported,
}

/// The `isa<>` chain at `:561`, `:667`, `:749` and `:797`, read once.
fn classify(op: &Op, current_value: Val) -> Arm {
    match op {
        Op::Sentient(sentient::Op::ScalarAdd { .. } | sentient::Op::ScalarSub { .. }) => {
            Arm::AddOrSub
        }
        Op::Sentient(
            sentient::Op::ReceiveAndStore { .. }
            | sentient::Op::LoadAndSend { .. }
            | sentient::Op::LoadComputeAndSend { .. },
        ) => Arm::Transfer(Some(MutableAddr::Only)),
        // `result_idx > 0 ? dst : src` (`:684-688`) — result 0 is the `src_res` address.
        Op::Sentient(sentient::Op::LoadAndStore { results, .. }) => Arm::Transfer(Some(
            if current_value == results.0 {
                MutableAddr::Src
            } else {
                MutableAddr::Dst
            },
        )),
        Op::Sentient(sentient::Op::LoadAndExtractScalar { addr_result, .. }) => {
            Arm::Transfer((current_value == *addr_result).then_some(MutableAddr::Only))
        }
        Op::Sentient(sentient::Op::For { carried, .. }) => carried
            .iter()
            .position(|value| value.result == current_value)
            .map_or(Arm::Unsupported, Arm::Loop),
        Op::UniformRegions(UniformRegions::UniformizeRegions { .. })
        | Op::Uniform(uniform::Op::UniformizeRegions { .. }) => Arm::Uniformize,
        _ => Arm::Unsupported,
    }
}

impl SsaMap {
    /// `ssa_negated_map_.at(value)` AS A SIGN — `s0` and `s2` (`:586-587`, `:717-718`, `:763-764`).
    /// `std::map::at` THROWS on a value `mapAllValues` never mapped, which is this `panic!`.
    fn sign_of(&self, value: Val) -> i64 {
        let Some(negated) = self.negated.get(&value) else {
            panic!(
                "ssa_negated_map_.at({value:?}) has no entry, where `std::map::at` throws \
                 (LiveRangeReduction.cpp:586-587)"
            )
        };
        if *negated { -1 } else { 1 }
    }

    /// THE TWO `ssa_expr_const_map_.at()` LOOKUPS EVERY ARM MAKES, plus the `DT_CHECK_MSG` that they
    /// are one length — *"Neither dominant_value, current_value are expected to be global."*
    fn const_offset_pair(&self, dominant: Val, other: Val) -> (Vec<i64>, Vec<i64>) {
        let (Some(dominant_consts), Some(other_consts)) = (
            self.const_offsets.get(&dominant),
            self.const_offsets.get(&other),
        ) else {
            panic!(
                "ssa_expr_const_map_.at() has no entry for {dominant:?} or {other:?}, where \
                 `std::map::at` throws (LiveRangeReduction.cpp:592-595)"
            )
        };
        if dominant_consts.len() != other_consts.len() {
            panic!(
                "DT_CHECK_MSG(Neither dominant_value, current_value are expected to be global.) \
                 (`LiveRangeReduction.cpp:596-598`): {} against {}",
                dominant_consts.len(),
                other_consts.len()
            )
        }
        (dominant_consts.clone(), other_consts.clone())
    }

    /// THE ALL-DIFFS-ZERO TEST the transfer and loop arms share (`:727-743`, `:775-791`) — a unit
    /// irresolvable for BOTH is skipped, for ONE it refuses, and any non-zero difference would need an
    /// extra instruction and so is refused on profitability.
    fn all_offsets_agree(&self, dominant: Val, other: Val, sign: i64) -> bool {
        let (dominant_consts, other_consts) = self.const_offset_pair(dominant, other);
        for (dominant_const, current_const) in dominant_consts.iter().zip(&other_consts) {
            if *dominant_const == i64::MAX && *current_const == i64::MAX {
                continue;
            }
            if *dominant_const == i64::MAX || *current_const == i64::MAX {
                return false;
            }
            if current_const.wrapping_sub(sign.wrapping_mul(*dominant_const)) != 0 {
                return false;
            }
        }
        true
    }

    /// Replaces: e502_reconstructOperation
    ///
    /// Rewrites the op binding `classes[class_index].values[current_value_index]` to read the
    /// DOMINATING value of its class instead. ⛔ ONLY THE SCALAR ARM MAY BUILD — it rebuilds around a
    /// fresh per-unit constant, while the five transfers and the loop replace one address operand and
    /// refuse any non-zero difference (`:733-737`, `:781-785`), an extra instruction there costing
    /// more than the register saved.
    ///
    /// ⛔ THE CLASS IS NAMED BY INDEX because the reference writes back through `ssa_value_list_[i]`
    /// (`:546`, `:630`), and `op_to_be_erased_` is a pass member (`:178`) e598 drains (`:1308`).
    pub fn reconstruct_operation(
        &mut self,
        class_index: usize,
        current_value_index: usize,
        dominant_value_index: usize,
        unit_body: &mut Vec<Op>,
        parent: &ParentRegionQuery,
        index_map: &impl UnitIndexMap,
        vals: &mut Values,
        to_be_erased: &mut Vec<Val>,
    ) -> Reconstruction {
        let Some(class) = self.classes.get(class_index) else {
            return Reconstruction::Done;
        };
        let (Some(current_value), Some(dominant_value)) = (
            class.values.get(current_value_index).copied(),
            class.values.get(dominant_value_index).copied(),
        ) else {
            return Reconstruction::Done;
        };
        let scope = unit_body.clone();
        let regions: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&regions);
        let Some(op) = defs.of(current_value) else {
            panic!(
                "reconstructOperation was asked about {current_value:?}, which has no defining op, \
                 where the reference dereferences a null Operation* \
                 (LiveRangeReduction.cpp:549-556)"
            )
        };
        // check if dominant_value is already used in the curren op — the check that prevents
        // switching iter_args' def-chain when iter_args.size > 1 (`:555-563`).
        if dialects::operands(op).contains(&dominant_value) {
            return Reconstruction::Done;
        }
        match classify(op, current_value) {
            Arm::AddOrSub => {
                let Op::Sentient(
                    sentient::Op::ScalarAdd {
                        ty,
                        reg,
                        element_size,
                        ..
                    }
                    | sentient::Op::ScalarSub {
                        ty,
                        reg,
                        element_size,
                        ..
                    },
                ) = op
                else {
                    return Reconstruction::Done;
                };
                let (ty, reg, element_size) = (*ty, *reg, *element_size);
                // The first operand that is a non-block-argument constant; `>= 2` is "none is", and
                // both constant makes live range reduction inapplicable anyway (`:566-580`).
                let reads = dialects::operands(op);
                let const_operand_idx = reads
                    .iter()
                    .position(|operand| is_sentient_constant(*operand, defs))
                    .unwrap_or(reads.len());
                if const_operand_idx >= 2 {
                    return Reconstruction::Done;
                }
                if !self.const_offsets.contains_key(&reads[1 - const_operand_idx]) {
                    return Reconstruction::Done;
                }
                let sign = self.sign_of(current_value) / self.sign_of(dominant_value);
                let (dominant_consts, current_consts) =
                    self.const_offset_pair(dominant_value, current_value);
                let mut new_const_val: Vec<i64> = Vec::with_capacity(dominant_consts.len());
                let mut common_val = i64::MAX;
                let mut are_all_val_equal = true;
                for (dominant_const, current_const) in dominant_consts.iter().zip(&current_consts) {
                    if *dominant_const == i64::MAX && *current_const == i64::MAX {
                        // if both SSAs are irresolvable for a unit, record it and continue.
                        new_const_val.push(i64::MAX);
                    } else if *dominant_const != i64::MAX && *current_const != i64::MAX {
                        let value = current_const.wrapping_sub(sign.wrapping_mul(*dominant_const));
                        new_const_val.push(value);
                        // only initialize common_val once
                        if common_val == i64::MAX {
                            common_val = value;
                        }
                    } else {
                        return Reconstruction::Done;
                    }
                    let last = new_const_val[new_const_val.len() - 1];
                    if common_val != i64::MAX && last != i64::MAX && last != common_val {
                        are_all_val_equal = false;
                    }
                }

                // remove the old operation
                to_be_erased.push(current_value);

                // optimization to remove addOp with zero-constant operand (`:625-632`).
                if are_all_val_equal && common_val == 0 && sign == 1 {
                    dialects::replace_all_uses_with(unit_body, current_value, dominant_value);
                    self.classes[class_index].values[current_value_index] = dominant_value;
                    return Reconstruction::Done;
                }

                // create new constOp + addOp/SubOp — `OpBuilder builder(op)` inserts IN FRONT of it.
                let mut built: Vec<Op> = Vec::new();
                let const_result = if are_all_val_equal {
                    let result = vals.mint();
                    built.push(Op::Sentient(sentient::Op::ScalarConstant {
                        value: common_val,
                        result,
                        reg_locale: sentient::RegType::Imm,
                        ty,
                        is_symbol: false,
                    }));
                    result
                } else {
                    let map_and_query =
                        create_map_and_query(&new_const_val, ty, parent, index_map, defs, vals);
                    built.extend(map_and_query.ops);
                    map_and_query.result
                };
                let new_result = vals.mint();
                // `new_op->setAttrs(op->getAttrs())` (`:653`) — the register and the width travel.
                built.push(Op::Sentient(if sign > 0 {
                    sentient::Op::ScalarAdd {
                        lhs: dominant_value,
                        rhs: const_result,
                        result: new_result,
                        reg,
                        element_size,
                        ty,
                    }
                } else {
                    sentient::Op::ScalarSub {
                        lhs: const_result,
                        rhs: dominant_value,
                        result: new_result,
                        reg,
                        element_size,
                        ty,
                    }
                }));
                let Some(at) = utils::path_of(unit_body, current_value) else {
                    return Reconstruction::Done;
                };
                for op in built.into_iter().rev() {
                    utils::insert_at(unit_body, &at, op);
                }
                dialects::replace_all_uses_with(unit_body, current_value, new_result);
                // ⭐ `DenseMap::operator[]` ON BOTH SIDES (`:656-658`), so a current value that was
                // never mapped copies an EMPTY offset list and `false` rather than nothing.
                let offsets = self
                    .const_offsets
                    .get(&current_value)
                    .cloned()
                    .unwrap_or_default();
                self.const_offsets.insert(new_result, offsets);
                let negated = self.negated.get(&current_value).copied().unwrap_or_default();
                self.negated.insert(new_result, negated);
                // replace cuurent_value in the position of ssa_list by new_op
                self.classes[class_index].values[current_value_index] = new_result;
                Reconstruction::Done
            }
            Arm::Transfer(addr) => {
                let Some(addr) = addr else {
                    return Reconstruction::Done;
                };
                let Some(op_operand) = addr.read(op) else {
                    return Reconstruction::Done;
                };
                if !self.const_offsets.contains_key(&op_operand) {
                    return Reconstruction::Done;
                }
                // No need to reconstruct constant operand — ⭐ THE OP ITSELF (`:711-714`), not
                // `isConstant<>`: a `uniform.query_map` of constants does NOT stop this one.
                if matches!(
                    defs.of(op_operand),
                    Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
                ) {
                    return Reconstruction::Done;
                }
                let sign = self.sign_of(current_value) / self.sign_of(dominant_value);
                if !self.all_offsets_agree(dominant_value, op_operand, sign) {
                    return Reconstruction::Done;
                }
                let Some(at) = utils::path_of(unit_body, current_value) else {
                    return Reconstruction::Done;
                };
                if let Some(target) = at.op_mut(unit_body) {
                    addr.write(target, dominant_value);
                }
                Reconstruction::Done
            }
            Arm::Loop(position) => {
                let Op::Sentient(sentient::Op::For { carried, .. }) = op else {
                    return Reconstruction::Done;
                };
                let Some(op_operand) = carried.get(position).map(|value| value.init) else {
                    return Reconstruction::Done;
                };
                // ⭐ THIS ARM TESTS THE CONSTANT BEFORE THE MAP, the transfer arm after (`:754-762`).
                if matches!(
                    defs.of(op_operand),
                    Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
                ) {
                    return Reconstruction::Done;
                }
                if !self.const_offsets.contains_key(&op_operand) {
                    return Reconstruction::Done;
                }
                let sign = self.sign_of(current_value) / self.sign_of(dominant_value);
                if !self.all_offsets_agree(dominant_value, op_operand, sign) {
                    return Reconstruction::Done;
                }
                let Some(at) = utils::path_of(unit_body, current_value) else {
                    return Reconstruction::Done;
                };
                if let Some(Op::Sentient(sentient::Op::For { carried, .. })) =
                    at.op_mut(unit_body)
                    && let Some(value) = carried.get_mut(position)
                {
                    value.init = dominant_value;
                }
                Reconstruction::Done
            }
            Arm::Uniformize => Reconstruction::Done,
            Arm::Unsupported => Reconstruction::Unsupported,
        }
    }
}

impl SsaMap {
    /// Replaces: e503_mapAllValues
    ///
    /// Files every register-related value of one program unit into the map, in the pre-order the
    /// reference's own comment calls topological, and hands back the `unit_name_to_index_map` it
    /// copies from the analysis (`:830`) for [`create_map_and_query`] to key its mapping by.
    ///
    /// ⛔ THE **REVERSE** SCANS (`:836-840`, `:861-864`) DECIDE which value of a class dominates
    /// another, so they are the port's content and not its style. ⛔ A YIELD MAPS ITS **PARENT'S**
    /// results, and only for the two parents whose `dyn_cast` succeeds (`:848-850`, `:865-868`).
    pub fn map_all_values<A: PropagationAnalysis>(
        &mut self,
        analysis: &mut A,
        unit_body: &[Op],
    ) -> A::Units {
        // copy unit_name_index_map from expression propagation
        let index_map = analysis.unit_index_map();
        self.map_block(analysis, unit_body, None, &[]);
        index_map
    }

    /// One block of the `walk<WalkOrder::PreOrder>` — the op, then its regions, which is what makes
    /// the order topological.
    fn map_block(
        &mut self,
        analysis: &mut impl PropagationAnalysis,
        block: &[Op],
        parent: Option<&Op>,
        enclosing: &[&[Op]],
    ) {
        let mut regions: Vec<&[Op]> = Vec::with_capacity(enclosing.len() + 1);
        regions.push(block);
        regions.extend_from_slice(enclosing);
        for op in block {
            self.map_operation(analysis, op, parent, Definitions::from_innermost(&regions));
            for region in dialects::regions_ref(op) {
                self.map_block(analysis, region, Some(op), &regions);
            }
        }
    }

    /// The four `isa<>` arms of the walk body (`:831-873`).
    fn map_operation(
        &mut self,
        analysis: &mut impl PropagationAnalysis,
        op: &Op,
        parent: Option<&Op>,
        defs: Definitions<'_>,
    ) {
        match op {
            Op::Sentient(sentient::Op::For { carried, .. }) => {
                // scan iter_args in reserve order to preferably shorten the liveRange of the first
                // iter_arg, which a kernel block load uses first.
                for value in carried.iter().rev() {
                    self.add_to_map(analysis, value.arg, defs);
                }
                // map constant operands — the eight kernel block loads with constant starting
                // addresses this exists for (`:842-847`).
                for operand in dialects::operands(op) {
                    if is_sentient_constant(operand, defs) {
                        self.add_to_map(analysis, operand, defs);
                    }
                }
            }
            Op::Sentient(sentient::Op::Yield { .. }) => {
                if let Some(Op::Sentient(sentient::Op::For { carried, .. })) = parent {
                    for value in carried.clone() {
                        self.add_to_map(analysis, value.result, defs);
                    }
                }
            }
            Op::Sentient(
                sentient::Op::ScalarAdd { .. }
                | sentient::Op::ScalarSub { .. }
                | sentient::Op::ReceiveAndStore { .. }
                | sentient::Op::LoadAndSend { .. }
                | sentient::Op::LoadAndStore { .. }
                | sentient::Op::LoadAndExtractScalar { .. }
                | sentient::Op::LoadComputeAndSend { .. },
            ) => {
                // scan reversely for the same reason described above.
                for result in dialects::results(op).into_iter().rev() {
                    self.add_to_map(analysis, result, defs);
                }
            }
            Op::Uniform(uniform::Op::Yield { .. }) => {
                if let Some(Op::UniformRegions(UniformRegions::UniformizeRegions {
                    results, ..
                })) = parent
                {
                    for result in results.clone() {
                        self.add_to_map(analysis, result, defs);
                    }
                }
            }
            // ⭐ THE WALK'S OWN FALL-THROUGH: it names seven ops of a dialect with twenty-nine.
            _ => {}
        }
    }
}

/// `is_any_of(.., SentientRegType::lrf, SentientRegType::jcr)` (`:1186-1188`, `:1199-1203`) — the two
/// register files a local region will hand a value out through.
fn is_yieldable_locale(locale: sentient::RegType) -> bool {
    matches!(locale, sentient::RegType::Lrf | sentient::RegType::Jcr)
}

/// `curr_op.hasAttr("regLocale")` AND ITS VALUE (`:1182-1186`) — the nine ops that declare the
/// SINGULAR attribute (`SentientOps.td:473, 521, 567, 681, 705, 806, 821, 834, 852`), `None` for the
/// rest, all of which bind exactly one result so the reference's default `result_idx = 0` is theirs.
///
/// ⭐ THE `Option` FIELDS **ARE** THE `hasAttr` ANSWER: `regLocale` is a `DefaultValuedAttr` on all
/// nine, so an op no allocator has spoken for carries none — which this island already spells as
/// `reg: Option<Reg>` on the two adds and `reg_locale: Option<RegType>` on `scalar_mul`.
fn single_reg_locale(op: &Op) -> Option<sentient::RegType> {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { reg, .. }
            | sentient::Op::ReceiveAndStore { reg, .. }
            | sentient::Op::LoadComputeAndSend { reg, .. }
            | sentient::Op::ReceiveAndExtractScalar { reg, .. }
            | sentient::Op::ScalarCopy { reg, .. },
        ) => Some(reg.locale),
        Op::Sentient(sentient::Op::ScalarAdd { reg, .. } | sentient::Op::ScalarSub { reg, .. }) => {
            reg.map(|reg| reg.locale)
        }
        Op::Sentient(sentient::Op::ScalarMul { reg_locale, .. }) => *reg_locale,
        Op::Sentient(sentient::Op::ScalarConstant { reg_locale, .. }) => Some(*reg_locale),
        _ => None,
    }
}

/// `regLocales[result_idx + curr_op.getNumOperands()]` (`:1197-1203`) — `None` where the op declares
/// no such array beside a `sentient.for` or `sentient.if`, which is the `hasAttr("regLocales")` and
/// the `isa<ForOp, IfOp>` the reference tests together.
///
/// ⛔ TRAP: THE REFERENCE'S OWN INDEX IS RIGHT ONLY FOR A LOOP. A `sentient.for` has `1 + n` operands
/// against a `1 + 2n` array, so `result_idx + numOperands` lands in the result half; a `sentient.if`
/// has TWO operands against an array of `numResults`, so the same expression runs PAST THE END and
/// `getValueRegLocale` has no `If` arm to disagree with it. Indexing by result alone is what both
/// layouts mean — see [`sentient::Carried::reg`] and [`sentient::Yielded::reg`].
fn result_reg_locale(op: &Op, result_idx: usize) -> Option<sentient::RegType> {
    match op {
        Op::Sentient(sentient::Op::For { carried, .. }) => {
            carried.get(result_idx).map(|value| value.reg.locale)
        }
        Op::Sentient(sentient::Op::If { yielded, .. }) => {
            yielded.get(result_idx).map(|value| value.reg.locale)
        }
        _ => None,
    }
}

/// Replaces: e504_optimizeUniformRegionYieldedValues
///
/// Grows every ONE-REGION `uniform.uniformize_regions` in `body` by one result for each value its
/// region computes into an `lrf` or `jcr` register and then never reads — a `sentient.for` or
/// `sentient.if` contributing one per such result rather than one per op.
///
/// ⛔ THE OP IS RE-READ FROM ITS SLOT EVERY TIME because [`add_result_to_yield`] answers with a WHOLE
/// NEW OP whose region-0 values are re-minted, and ⭐ `ops_to_be_delected` (`:1208-1209`) IS that slot
/// assignment. ⛔ `parent` is REPLACED on the way into a local region, where
/// `getQueryKeyAndUnitsFromParentRegion` reads exactly its two fields (`Sentient/Utils.cpp:53-54`).
pub fn optimize_uniform_region_yielded_values(
    body: &mut Vec<Op>,
    parent: &ParentRegionQuery,
    enclosing: &[&[Op]],
    vals: &mut Values,
) {
    for index in 0..body.len() {
        if dialects::local_region_count(&body[index]) == Some(1) {
            grow_yielded_values(body, index, parent, enclosing, vals);
        }
        if dialects::regions_ref(&body[index]).is_empty() {
            continue;
        }
        // The walk descends AFTER the op itself, and against the block as the rewrite left it.
        let snapshot = body.clone();
        let mut scopes: Vec<&[Op]> = Vec::with_capacity(enclosing.len() + 1);
        scopes.push(&snapshot);
        scopes.extend_from_slice(enclosing);
        match &mut body[index] {
            Op::UniformRegions(regions) => {
                for region in regions.regions_mut() {
                    let inner = ParentRegionQuery {
                        key: region.arg,
                        units: region.units.clone(),
                    };
                    optimize_uniform_region_yielded_values(
                        &mut region.body,
                        &inner,
                        &scopes,
                        vals,
                    );
                }
            }
            op => {
                for region in dialects::regions_mut(op) {
                    optimize_uniform_region_yielded_values(region, parent, &scopes, vals);
                }
            }
        }
    }
}

/// The `region_num`/`op_num` pair of loops (`:1173-1206`) over ONE one-region op, whose slot the
/// grown op is written back into. `region_num` is only ever 0 there, which is what makes
/// `addResultToYield`'s own `DT_CHECK(r == 1)` hold.
fn grow_yielded_values(
    body: &mut Vec<Op>,
    index: usize,
    parent: &ParentRegionQuery,
    enclosing: &[&[Op]],
    vals: &mut Values,
) {
    let Some(Op::UniformRegions(op)) = body.get(index) else {
        return;
    };
    let Some(num_ops) = op.regions().first().map(|region| region.body.len()) else {
        return;
    };
    for op_num in 0..num_ops {
        let Some(Op::UniformRegions(op)) = body.get(index) else {
            return;
        };
        let Some(region) = op.regions().first() else {
            return;
        };
        let Some(curr_op) = region.body.get(op_num) else {
            continue;
        };
        // `curr_op.use_empty()` — every result unread, and a value bound inside a local region can
        // only be read inside it.
        if dialects::results(curr_op)
            .iter()
            .any(|result| dialects::use_count(*result, &region.body) != 0)
        {
            continue;
        }
        let positions: Vec<usize> = match single_reg_locale(curr_op) {
            // `hasAttr("regLocale")` — one register for the whole op, hence result 0 alone.
            Some(locale) if is_yieldable_locale(locale) => vec![0],
            Some(_) => Vec::new(),
            None => (0..dialects::results(curr_op).len())
                .filter(|at| result_reg_locale(curr_op, *at).is_some_and(is_yieldable_locale))
                .collect(),
        };
        for result_idx in positions {
            let snapshot = body.clone();
            let mut scopes: Vec<&[Op]> = Vec::with_capacity(enclosing.len() + 1);
            scopes.push(&snapshot);
            scopes.extend_from_slice(enclosing);
            let Some(Op::UniformRegions(op)) = snapshot.get(index) else {
                return;
            };
            let grown = add_result_to_yield(
                op,
                parent,
                0,
                utils::InBlock(op_num),
                result_idx,
                &scopes,
                vals,
            );
            body[index] = Op::UniformRegions(grown);
        }
    }
}

/// `op->emitError("Failed in reconstructing the following operation!")` (`:906`) — ⛔ THE MESSAGE
/// VERBATIM, exclamation mark included.
const FAILED_RECONSTRUCTION: &str = "Failed in reconstructing the following operation!";

/// `op->emitError(..); op->dump(); signalPassFailure();` (`:906-908`) AS DATA — one value whose
/// defining op [`SsaMap::reconstruct_operation`] had no arm for.
///
/// ⭐ NOT A `Result`: the reference records the failure and CARRIES ON to the next value of the same
/// class (`:905-909`), so the pass's verdict is the accumulated list. Same shape as
/// [`crate::transform::sentient::register_allocation`]'s `UnknownLocale`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructionFailure {
    /// The value whose op was not rewritten — an identity, not a borrow.
    pub at: Val,
    /// `op->dump()`, into a buffer a caller can read rather than the process's stderr.
    pub dumped: String,
    /// The message the reference emitted.
    pub message: &'static str,
}

impl SsaMap {
    /// Replaces: e560_reduceLiveRange
    ///
    /// Per equivalence class, from its SECOND value on: point each value's op at the closest earlier
    /// value of the class that dominates it, recording the ops no arm of the rewriter covers.
    ///
    /// ⛔ TRAP: `PropagationAnalysis& expr_prop` IS UNREAD IN THE BODY (`:885-911`) and is dropped,
    /// as is `DominanceInfo dominance_info(unit_op)` (`:889`) — [`properly_dominates`] answers from
    /// `unit_body` itself, which is the positioning mechanism a port supplies.
    /// ⛔ INDEX 0 IS NEVER RECONSTRUCTED (`:895`): it is the one every other value of the class moves
    /// onto. ⭐ `isa<BlockArgument>` IS "no defining op in this unit" — an iter arg, skipped (`:899`).
    pub fn reduce_live_range(
        &mut self,
        check: ElementSizeCheck,
        unit_body: &mut Vec<Op>,
        parent: &ParentRegionQuery,
        index_map: &impl UnitIndexMap,
        vals: &mut Values,
        to_be_erased: &mut Vec<Val>,
    ) -> Vec<ReconstructionFailure> {
        let mut failures = Vec::new();
        for class_index in 0..self.classes.len() {
            let mut current_value_index = 1;
            // `ssa_list.size()` is re-read every round, because `reconstructOperation` writes back
            // through `ssa_value_list_[i]` (`:546`, `:630`).
            while let Some(ssa_list) = self
                .classes
                .get(class_index)
                .map(|class| class.values.clone())
                .filter(|values| current_value_index < values.len())
            {
                let current = ssa_list[current_value_index];
                let scope = unit_body.clone();
                let regions: [&[Op]; 1] = [&scope];
                let defs = Definitions::from_innermost(&regions);
                if defs.of(current).is_none() {
                    current_value_index += 1;
                    continue;
                }
                let dominant = find_dominant_value(&ssa_list, current_value_index, check, &scope, defs);
                let mut dumped = String::new();
                print_value(current, defs, &mut dumped);
                if let Some(dominant_value_index) = dominant
                    && self.reconstruct_operation(
                        class_index,
                        current_value_index,
                        dominant_value_index,
                        unit_body,
                        parent,
                        index_map,
                        vals,
                        to_be_erased,
                    ) == Reconstruction::Unsupported
                {
                    failures.push(ReconstructionFailure {
                        at: current,
                        dumped,
                        message: FAILED_RECONSTRUCTION,
                    });
                }
                current_value_index += 1;
            }
        }
        failures
    }
}

/// `-dcc-live-range-reduction-disable`, `cl::init(false)` (`:103-105`) — a `dcc-opt` flag, not a
/// program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `opts_.OptLevel == 0` (`:1292`) — the PIPELINE's optimisation level, `2` by default
/// (`dcc/tools/Options/dcc-pass-option.h:63-65`), so the shipped pipeline never asks the estimator.
const OPT_LEVEL_ZERO: bool = false;

/// `module_op->emitError("Unable to perform expression propagation"); signalPassFailure();`
/// (`:1279-1281`) AS DATA.
///
/// ⛔⛔ AND THE PASS CARRIES ON: there is no `return` under those two statements, so the walk below
/// runs on expressions the analysis itself disowned. A `Result` here would invent an early exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropagationFailure;

/// Replaces: e598_runOnOperation
///
/// The pass entry: every one-region `uniform.uniformize_regions` grows the register values its region
/// drops, then each program unit's values are mapped and their live ranges reduced, and the ops the
/// reductions orphaned are erased once at the end (`:1267-1310`).
///
/// ⛔ THE FOUR `clear()`s (`:1296-1299`) ARE A FRESH [`SsaMap`] PER UNIT — the map is per-unit state
/// the reference keeps in the pass, and carrying one unit's classes into the next would merge
/// registers across units. ⛔ THE ERASE QUEUE IS DRAINED ONCE FOR THE WHOLE MODULE (`:1306-1308`).
/// ⛔ `unit_keys` IS A PARAMETER: [`ProgramUnit`] does not carry its region argument, the same seam
/// [`ParentRegionQuery`] already is.
pub fn run_on_operation<A: Arch, M: Model, W: Workload, P: PropagationAnalysis>(
    program: &mut Program<A, M, W>,
    analysis: &mut P,
    unit_keys: &[Val],
    vals: &mut Values,
) -> (Option<PropagationFailure>, Vec<ReconstructionFailure>) {
    if DISABLE_THIS_PASS {
        return (None, Vec::new());
    }
    // ⛔ DIVERGENCE: `optimizeUniformRegionYieldedValues(module_op)` walks the WHOLE module, and this
    // walks each program unit's body — which is where e504's `getQueryKeyAndUnitsFromParentRegion`
    // has a program unit to read a key and a unit list out of (`Sentient/Utils.cpp:44-54`).
    for (unit, key) in program.units.iter_mut().zip(unit_keys.iter()) {
        let parent = ParentRegionQuery {
            key: *key,
            units: unit.on.vals(),
        };
        optimize_uniform_region_yielded_values(&mut unit.body, &parent, &[], vals);
    }
    let propagation = if analysis.is_propagation_successful() {
        None
    } else {
        Some(PropagationFailure)
    };

    let mut failures = Vec::new();
    let mut to_be_erased: Vec<Val> = Vec::new();
    for (unit, key) in program.units.iter_mut().zip(unit_keys.iter()) {
        // `std::set<SenComponents> list = {PT, PE, SFP}; skip_elem_size_ = list.count(type) > 0`
        // (`:1286-1290`) — ⭐ `getUnitType` IS THE GENERIC COMPONENT, so one PT row names them all.
        let check = match unit.on.kind().generic() {
            GenericComp::Pt | GenericComp::Pe | GenericComp::Sfp => ElementSizeCheck::Skipped,
            _ => ElementSizeCheck::Required,
        };
        // `getChildAnalysis<InstructionEstimator>(unit_op)` IS CONSTRUCTED PER UNIT (`:1291-1292`).
        let mut instruction_estimator = OutOfScopeInstructionEstimator;
        if OPT_LEVEL_ZERO && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        let parent = ParentRegionQuery {
            key: *key,
            units: unit.on.vals(),
        };
        let mut map = SsaMap::default();
        let index_map = map.map_all_values(analysis, &unit.body);
        failures.extend(map.reduce_live_range(
            check,
            &mut unit.body,
            &parent,
            &index_map,
            vals,
            &mut to_be_erased,
        ));
    }
    // `for (auto op : op_to_be_erased_) op->erase();` — ⭐ EVERY [`Val`] IS UNIQUE ACROSS THE MODULE,
    // so the unit a queued value came from is the only one an erase can find it in.
    for val in to_be_erased {
        for unit in program.units.iter_mut() {
            erase_defining_op(&mut unit.body, val);
        }
    }
    (propagation, failures)
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Elements};
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::DfirUnit;
    use crate::islands::sentient::dialects::sentient::{Extent, Reg, RegType};

    /// A `sentient.load_and_extract_scalar` reading `addr`, `size` bits at a time, into `locale`.
    fn extract(addr: Val, results: (Val, Val), size: u32, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::LoadAndExtractScalar {
            mutable_addr: addr,
            immutable_addr: Val(0),
            increment: Val(0),
            consumer: SendEnd::to_self(Val(0)),
            addr_result: results.0,
            data_result: results.1,
            total_elements: Elements(8),
            element_size: Bits(size),
            addr_reg: Reg {
                locale,
                index: None,
            },
            data_reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            dbg_name: None,
        })
    }

    fn constant(result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 64,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    /// e056 — all three fields are replaced, not merged.
    #[test]
    fn set_replaces_every_field_including_the_constraint_system() {
        let mut info = ExprInfo {
            expr_coeffs: vec![9, 9, 9],
            args: vec![Val(1), Val(2), Val(3)],
            local_vars_constraints: FlatAffineValueConstraints(1),
        };
        info.set(vec![1, 2], vec![Val(7)], FlatAffineValueConstraints(2));
        assert_eq!(info.expr_coeffs, vec![1, 2]);
        assert_eq!(info.args, vec![Val(7)]);
        assert_eq!(info.local_vars_constraints.0, 2);
    }

    /// e058 — equal known widths match; an op with no `element_size` (the reference's `-1`) does not.
    #[test]
    fn element_sizes_match_only_when_both_are_known_and_equal() {
        let ops = vec![
            extract(Val(1), (Val(2), Val(3)), 16, RegType::Lar),
            extract(Val(4), (Val(5), Val(6)), 16, RegType::Lar),
            extract(Val(7), (Val(8), Val(9)), 8, RegType::Lbr),
            constant(Val(10)),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert!(are_element_size_identical(Val(2), Val(5), defs));
        assert!(!are_element_size_identical(Val(2), Val(8), defs));
        assert!(!are_element_size_identical(Val(2), Val(10), defs));
    }

    /// e060 — the same register file matches, two different ones do not.
    #[test]
    fn reglocales_match_per_register_file() {
        let ops = vec![
            extract(Val(1), (Val(2), Val(3)), 16, RegType::Lar),
            extract(Val(4), (Val(5), Val(6)), 16, RegType::Lar),
            extract(Val(7), (Val(8), Val(9)), 16, RegType::Lbr),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert!(are_reglocales_matching(Val(2), Val(5), defs));
        assert!(!are_reglocales_matching(Val(2), Val(8), defs));
    }

    /// e059 — the walk crosses the offsetting adds and stops on the `uniform.query_map`, taking the
    /// operand that is NOT a constant.
    #[test]
    fn ancestor_walk_skips_constant_offsets_and_stops_at_the_query_map() {
        let ops = vec![
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(1),
                pairs: vec![(Val(30), Val(31))],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(2),
                map: Val(1),
                key: Val(0),
            }),
            constant(Val(3)),
            // `%4 = %2 + %3` — the constant is the rhs, so the walk follows the lhs.
            add(Val(2), Val(3), Val(4)),
            // `%5 = %3 + %4` — the constant is the LHS, so the walk follows the rhs.
            add(Val(3), Val(4), Val(5)),
            extract(Val(5), (Val(6), Val(7)), 16, RegType::Lar),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert_eq!(
            get_first_global_or_constant_ancestor(Val(6), &ops, defs),
            Some(Val(2))
        );
    }

    /// e059 — a value defined by an op with no address operand is the `emitOpError` that fails the
    /// pass, and a `load_and_store` result walks the side it came out of.
    #[test]
    fn ancestor_walk_splits_load_and_store_by_result_and_refuses_the_rest() {
        let ops = vec![
            constant(Val(1)),
            constant(Val(2)),
            Op::Sentient(sentient::Op::ScalarCopy {
                input: Val(1),
                result: Val(3),
                reg: Reg {
                    locale: RegType::Lar,
                    index: None,
                },
                program_header: false,
                element_size: None,
            }),
        ];
        let mut ops = ops;
        ops.push(Op::Sentient(sentient::Op::LoadAndStore {
            src: Val(20),
            dst: Val(21),
            src_mutable_addr: Val(1),
            src_immutable_addr: Val(0),
            src_inc: Val(0),
            dst_mutable_addr: Val(2),
            dst_immutable_addr: Val(0),
            dst_inc: Val(0),
            multicast_info: None,
            results: (Val(4), Val(5)),
            extent: Extent::of(Elements(8), Bits(16)),
            stride: 1,
            rotate_val: None,
            shuffle_mode: sentient::ShuffleMode::NoShuffle,
            src_reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dst_reg: Reg {
                locale: RegType::Lbr,
                index: None,
            },
            dir: None,
            is_ibr_write: false,
            dbg_name: None,
        }));
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        // `sentient.scalar_copy` has no arm — `emitOpError("can't find global ancestor")`.
        assert_eq!(get_first_global_or_constant_ancestor(Val(3), &ops, defs), None);
        // A constant IS the ancestor, and answers before any region test.
        assert_eq!(
            get_first_global_or_constant_ancestor(Val(1), &ops, defs),
            Some(Val(1))
        );
        // ⭐ EACH `load_and_store` RESULT WALKS ITS OWN SIDE — the divergence this port's doc names.
        assert_eq!(
            get_first_global_or_constant_ancestor(Val(4), &ops, defs),
            Some(Val(1))
        );
        assert_eq!(
            get_first_global_or_constant_ancestor(Val(5), &ops, defs),
            Some(Val(2))
        );
    }
    /// A `PropagationAnalysis` that answers with exactly what the fixture put in it.
    struct Canned {
        map: ExprInfoMap,
        flats: BTreeMap<u32, FlattenedExpr>,
    }

    impl PropagationAnalysis for Canned {
        type Units = OutOfScopeUnitIndexMap;

        fn affine_expression(&mut self, _val: Val) -> ExprInfoMap {
            self.map.clone()
        }

        fn flattened_affine_expr(&self, map: PropagatedMap) -> FlattenedExpr {
            self.flats[&map.id].clone()
        }

        fn unit_index_map(&self) -> Self::Units {
            OutOfScopeUnitIndexMap
        }
    }

    /// One resolved bucket over `num_dims` dimensions.
    fn bucket(id: u32, num_dims: usize, args: Vec<Val>) -> Option<PropagatedExpr> {
        Some(PropagatedExpr {
            propagated_map: PropagatedMap { id, num_dims },
            propagated_args: args,
            cannot_be_resolved: false,
        })
    }

    /// e304 — a mapped value's block, and the two spaces the reference puts before the newline.
    #[test]
    fn print_ssa_map_dumps_the_const_offsets_and_the_negated_flag_per_value() {
        let ops = vec![constant(Val(1)), constant(Val(2))];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);
        let map = SsaMap {
            classes: vec![EquivalenceClass {
                expr_info: ExprInfo::default(),
                values: vec![Val(1), Val(2)],
            }],
            const_offsets: BTreeMap::from([(Val(1), vec![4, 8])]),
            negated: BTreeMap::from([(Val(1), true)]),
        };

        let mut out = String::new();
        map.print_mapped_value(Val(1), defs, &mut out);
        assert!(
            out.ends_with("  \nconst offset: 4, 8, \nexpr negated: true\n"),
            "{out}"
        );
        // ⛔ A VALUE WITH NO ENTRY IS EMPTY AND `false`, not missing.
        let mut out = String::new();
        map.print_mapped_value(Val(2), defs, &mut out);
        assert!(
            out.ends_with("  \nconst offset: \nexpr negated: false\n"),
            "{out}"
        );
        // No classes, no output at all.
        let mut out = String::new();
        SsaMap::default().print_ssa_map(defs, &mut out);
        assert!(out.is_empty());
    }

    /// e305 — the sign normalisation, the popped constant, the null bucket's sentinel, and the
    /// failure two disagreeing buckets produce.
    #[test]
    fn get_base_expr_normalizes_the_sign_and_refuses_disagreeing_units() {
        let ops = vec![constant(Val(1))];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        let mut agreeing = Canned {
            // Unit 0 reads bucket 0; unit 1 reads the null bucket 1.
            map: ExprInfoMap {
                exprs: vec![bucket(0, 1, vec![Val(7)]), None],
                buckets: vec![0, 1],
            },
            flats: BTreeMap::from([(
                0,
                FlattenedExpr {
                    coeffs: vec![-2, 5],
                    constraints: FlatAffineValueConstraints(3),
                    num_local_vars: 0,
                },
            )]),
        };
        let base = get_base_expr(&mut agreeing, Val(1), defs).unwrap();
        // ⭐ THE CONSTANT IS POPPED OFF THE BACK, and the remaining coefficient is sign-flipped.
        assert_eq!(base.expr_info.expr_coeffs, vec![2]);
        assert!(base.negated);
        assert_eq!(base.expr_info.args, vec![Val(7)]);
        assert_eq!(base.const_val, vec![5, i64::MAX]);

        // Two resolved buckets whose coefficients differ — `LogicalResult::failure()`.
        let mut disagreeing = Canned {
            map: ExprInfoMap {
                exprs: vec![bucket(0, 1, vec![Val(7)]), bucket(1, 1, vec![Val(7)])],
                buckets: vec![0, 1],
            },
            flats: BTreeMap::from([
                (
                    0,
                    FlattenedExpr {
                        coeffs: vec![2, 5],
                        constraints: FlatAffineValueConstraints(3),
                        num_local_vars: 0,
                    },
                ),
                (
                    1,
                    FlattenedExpr {
                        coeffs: vec![3, 5],
                        constraints: FlatAffineValueConstraints(3),
                        num_local_vars: 0,
                    },
                ),
            ]),
        };
        assert!(get_base_expr(&mut disagreeing, Val(1), defs).is_none());
    }

    /// e306 — the walk crosses the constant offset, the loop carry and the address result, and stops
    /// dead on the data result.
    #[test]
    fn root_iter_arg_chains_addresses_and_stops_on_a_data_result() {
        let ops = vec![
            constant(Val(1)),
            // `%3 = %1 + %2` — `$inp1` is the constant, so the walk follows `$inp2`, a block argument.
            add(Val(1), Val(2), Val(3)),
            extract(Val(3), (Val(4), Val(5)), 16, RegType::Lar),
            Op::Sentient(sentient::Op::For {
                iv: Val(60),
                bound: Val(1),
                bound_reg: None,
                carried: vec![sentient::Carried {
                    init: Val(4),
                    arg: Val(61),
                    result: Val(62),
                    reg: Reg {
                        locale: RegType::Unknown,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert_eq!(get_root_iter_arg(Val(4), defs), Val(2));
        // ⛔ THE DATA RESULT CHAINS NOWHERE.
        assert_eq!(get_root_iter_arg(Val(5), defs), Val(5));
        // A loop result walks back to the `initArgs` entry it came from, then on down the chain.
        assert_eq!(get_root_iter_arg(Val(62), defs), Val(2));
    }

    /// e307 — the query is re-minted inside region 1 over region 1's own units, and a mapping that
    /// misses one of them refuses.
    #[test]
    fn clone_value_to_region_remints_the_query_over_the_regions_own_units() {
        let mut vals = Values::default();
        let unit_a = vals.mint();
        let unit_b = vals.mint();
        let val_a = vals.mint();
        let val_b = vals.mint();
        let mapping = vals.mint();
        let query = vals.mint();
        let r0_arg = vals.mint();
        let r1_arg = vals.mint();

        let covering = |pairs: Vec<(Val, Val)>| {
            UniformRegions::UniformizeRegions {
                regions: vec![
                    LocalRegion {
                        arg: r0_arg,
                        units: vec![unit_a, unit_b],
                        body: vec![
                            constant(val_a),
                            constant(val_b),
                            Op::Uniform(uniform::Op::DefImmutableMapping {
                                result: mapping,
                                pairs,
                            }),
                            Op::Uniform(uniform::Op::QueryMap {
                                result: query,
                                map: mapping,
                                key: r0_arg,
                            }),
                            Op::Uniform(uniform::Op::Yield {
                                operands: Vec::new(),
                            }),
                        ],
                    },
                    LocalRegion {
                        arg: r1_arg,
                        units: vec![unit_a, unit_b],
                        body: vec![Op::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        })],
                    },
                ],
                results: Vec::new(),
                yielded: Vec::new(),
            }
        };

        let mut op = covering(vec![(unit_a, val_a), (unit_b, val_b)]);
        let cloned =
            clone_value_to_region(query, SecondRegion::of(&mut op).unwrap(), &mut vals).unwrap();
        let region1 = &op.regions()[1];
        assert_eq!(region1.body.len(), 3);
        assert!(matches!(
            &region1.body[0],
            Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })
                if *pairs == vec![(unit_a, val_a), (unit_b, val_b)]
        ));
        // ⭐ THE NEW QUERY READS REGION 1'S OWN ARGUMENT, which is what makes it usable there.
        assert!(matches!(
            &region1.body[1],
            Op::Uniform(uniform::Op::QueryMap { result, key, .. })
                if *result == cloned && *key == r1_arg
        ));

        // ⛔ A MAPPING THAT COVERS ONLY ONE OF THE TWO UNITS REFUSES, and nothing is inserted.
        let mut short = covering(vec![(unit_a, val_a)]);
        assert_eq!(
            clone_value_to_region(query, SecondRegion::of(&mut short).unwrap(), &mut vals),
            None
        );
        assert_eq!(short.regions()[1].body.len(), 1);
    }

    /// e440 — the second value of one expression joins the first's class, and a constant is mapped
    /// without ever entering one.
    #[test]
    fn add_to_map_classes_the_non_constants_and_only_maps_the_constant() {
        let ops = vec![
            constant(Val(1)),
            extract(Val(20), (Val(2), Val(3)), 16, RegType::Lar),
            extract(Val(21), (Val(5), Val(6)), 16, RegType::Lar),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);
        let mut canned = Canned {
            map: ExprInfoMap {
                exprs: vec![bucket(0, 1, vec![Val(7)])],
                buckets: vec![0],
            },
            flats: BTreeMap::from([(
                0,
                FlattenedExpr {
                    coeffs: vec![2, 5],
                    constraints: FlatAffineValueConstraints(3),
                    num_local_vars: 0,
                },
            )]),
        };

        let mut map = SsaMap::default();
        map.add_to_map(&mut canned, Val(2), defs);
        map.add_to_map(&mut canned, Val(5), defs);
        // ⛔ THE CONSTANT OPENS NO CLASS AND JOINS NONE.
        map.add_to_map(&mut canned, Val(1), defs);

        assert_eq!(map.classes.len(), 1);
        assert_eq!(map.classes[0].values, vec![Val(2), Val(5)]);
        assert_eq!(map.classes[0].expr_info.expr_coeffs, vec![2]);
        // ⭐ ALL THREE ARE MAPPED, the constant included.
        assert_eq!(map.const_offsets.get(&Val(1)), Some(&vec![5]));
        assert_eq!(map.const_offsets.get(&Val(5)), Some(&vec![5]));
        assert_eq!(map.negated.get(&Val(1)), Some(&false));
    }

    /// e441 — the closest dominator wins; a differing width refuses unless the unit waives the check;
    /// and two adds with different def-use roots are skipped rather than chained.
    #[test]
    fn find_dominant_value_takes_the_closest_matching_dominator() {
        let ops = vec![
            extract(Val(1), (Val(2), Val(3)), 16, RegType::Lar),
            extract(Val(2), (Val(4), Val(5)), 16, RegType::Lar),
            // Same locale, HALF the element width.
            extract(Val(4), (Val(6), Val(7)), 8, RegType::Lar),
            // `%10 = %2 + %11`, rooted through the first extract in the block argument `%1`.
            add(Val(2), Val(11), Val(10)),
            // `%13 = %12 + %11`, rooted in the block argument `%12`.
            add(Val(12), Val(11), Val(13)),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        let chain = [Val(2), Val(4), Val(6)];
        assert_eq!(
            find_dominant_value(&chain, 1, ElementSizeCheck::Required, &ops, defs),
            Some(0)
        );
        // ⛔ 8 BITS AGAINST 16 IS NO MATCH, and nothing earlier matches either.
        assert_eq!(
            find_dominant_value(&chain, 2, ElementSizeCheck::Required, &ops, defs),
            None
        );
        // ⭐ A PT/PE/SFP UNIT WAIVES THAT CHECK, so the immediate dominator answers.
        assert_eq!(
            find_dominant_value(&chain, 2, ElementSizeCheck::Skipped, &ops, defs),
            Some(1)
        );
        // ⛔ TWO ADDS WITH DIFFERENT ROOTS ARE SKIPPED — the `continue` at `:513`.
        assert_eq!(
            find_dominant_value(&[Val(10), Val(13)], 1, ElementSizeCheck::Skipped, &ops, defs),
            None
        );
    }

    /// A `dataflow.get_unit` standing for one unit handle, so that `collectUnitVals` recognises it.
    fn get_unit(result: Val) -> Op {
        Op::Dataflow(lower::dataflow::Op::GetUnit {
            result,
            residency: crate::units::Residency::Global,
            unit: crate::units::DfirUnit::Hbm,
            num_folds: None,
            reg_locale: None,
        })
    }

    /// e442 — a one-region op gains its second region on exactly the units region 0 does not cover,
    /// and the new result is yielded from BOTH sides: region 0 yields the op's own result, region 1 the
    /// ancestor query re-minted over its own unit.
    #[test]
    fn add_result_to_yield_opens_a_second_region_on_the_units_region_zero_leaves() {
        let mut vals = Values::default();
        let unit_a = vals.mint();
        let unit_b = vals.mint();
        let const_a = vals.mint();
        let const_b = vals.mint();
        let mapping = vals.mint();
        let query = vals.mint();
        let addr_result = vals.mint();
        let data_result = vals.mint();
        let r0_arg = vals.mint();

        let op = UniformRegions::UniformizeRegions {
            regions: vec![LocalRegion {
                arg: r0_arg,
                units: vec![unit_a],
                body: vec![
                    constant(const_a),
                    constant(const_b),
                    Op::Uniform(uniform::Op::DefImmutableMapping {
                        result: mapping,
                        pairs: vec![(unit_a, const_a), (unit_b, const_b)],
                    }),
                    Op::Uniform(uniform::Op::QueryMap {
                        result: query,
                        map: mapping,
                        key: r0_arg,
                    }),
                    extract(query, (addr_result, data_result), 16, RegType::Lar),
                    Op::Uniform(uniform::Op::Yield {
                        operands: Vec::new(),
                    }),
                ],
            }],
            results: Vec::new(),
            yielded: Vec::new(),
        };
        // ⭐ THE OP IS IN THE BLOCK IT IS REWRITTEN IN, which is how the widths and locales of values
        // inside its regions are readable from the enclosing scope at all.
        let enclosing = vec![
            get_unit(unit_a),
            get_unit(unit_b),
            Op::UniformRegions(op.clone()),
        ];
        let scopes: [&[Op]; 1] = [&enclosing];
        let parent = ParentRegionQuery {
            key: r0_arg,
            units: vec![unit_a, unit_b],
        };

        let grown = add_result_to_yield(
            &op,
            &parent,
            0,
            utils::InBlock(4),
            0,
            &scopes,
            &mut vals,
        );
        let UniformRegions::UniformizeRegions {
            regions,
            results,
            yielded,
        } = &grown
        else {
            panic!("addResultToYield answers with a uniform.uniformize_regions")
        };

        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].units, vec![unit_a]);
        // ⛔ THE SECOND REGION'S UNIT LIST IS THE PARENT'S MINUS REGION 0'S.
        assert_eq!(regions[1].units, vec![unit_b]);
        assert_eq!(results.len(), 1);
        // The width and register file are read from the ORIGINAL result, not the clone.
        assert_eq!(
            yielded,
            &vec![YieldedReg {
                element_size: Some(Bits(16)),
                locale: RegType::Lar,
            }]
        );

        // Region 0 yields its own clone of the op's result.
        let cloned_addr = dialects::results(&regions[0].body[4])[0];
        assert!(matches!(
            regions[0].body.last(),
            Some(Op::Uniform(uniform::Op::Yield { operands })) if *operands == vec![cloned_addr]
        ));
        // ⭐ REGION 1 YIELDS A QUERY IT MINTED ITSELF, over a mapping of its own single unit.
        assert!(matches!(
            &regions[1].body[0],
            Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. }) if pairs.len() == 1
        ));
        let Op::Uniform(uniform::Op::QueryMap {
            result: re_minted,
            key,
            ..
        }) = &regions[1].body[1]
        else {
            panic!("region 1 opens with the mapping and the query cloneValueToRegion inserted")
        };
        assert_eq!(*key, regions[1].arg);
        assert!(matches!(
            regions[1].body.last(),
            Some(Op::Uniform(uniform::Op::Yield { operands })) if *operands == vec![*re_minted]
        ));
    }

    /// e443 — one constant per unit index, keyed in `selectIndicesForUnits`' reversed order, and the
    /// `INT64_MAX` sentinel's slot produces no constant at all.
    #[test]
    fn create_map_and_query_keys_the_parents_units_in_reversed_order() {
        /// The unit's own value number as its index — the out-of-scope name lookup stood in for.
        struct ByValue;
        impl UnitIndexMap for ByValue {
            fn index_of(&self, unit: Val) -> UnitIndex {
                UnitIndex(unit.0)
            }
        }

        let scope = vec![get_unit(Val(0)), get_unit(Val(2))];
        let scopes: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&scopes);
        let mut vals = Values::default();
        let key = vals.mint();
        let parent = ParentRegionQuery {
            key,
            units: vec![Val(0), Val(2)],
        };

        // Index 1's offset is the sentinel, and no unit of the parent region claims it.
        let built = create_map_and_query(
            &[7, i64::MAX, 9],
            ScalarTy::Index,
            &parent,
            &ByValue,
            defs,
            &mut vals,
        );

        // ⛔ TWO CONSTANTS FOR THREE OFFSETS — the sentinel's slot is a null the reference pushes.
        assert_eq!(built.ops.len(), 4);
        let values: Vec<i64> = built.ops[..2]
            .iter()
            .map(|op| match op {
                Op::Sentient(sentient::Op::ScalarConstant {
                    value,
                    reg_locale: RegType::Imm,
                    ty: ScalarTy::Index,
                    ..
                }) => *value,
                other => panic!("the constants come first, not {other:?}"),
            })
            .collect();
        assert_eq!(values, vec![7, 9]);
        let seven = dialects::results(&built.ops[0])[0];
        let nine = dialects::results(&built.ops[1])[0];

        // ⭐ REVERSED: the LIFO pops `%2` before `%0`, so unit 2's constant is the first pair.
        let Op::Uniform(uniform::Op::DefImmutableMapping {
            result: mapping,
            pairs,
        }) = &built.ops[2]
        else {
            panic!("the mapping follows the constants")
        };
        assert_eq!(*pairs, vec![(Val(2), nine), (Val(0), seven)]);
        assert!(matches!(
            &built.ops[3],
            Op::Uniform(uniform::Op::QueryMap { result, map, key: queried })
                if *result == built.result && map == mapping && *queried == key
        ));
    }

    /// e502 — a zero difference collapses the add away, a non-zero one rebuilds it around a fresh
    /// constant, and a transfer op has its address operand rewritten in place instead.
    #[test]
    fn reconstruct_operation_collapses_a_zero_offset_and_rebuilds_a_non_zero_one() {
        let parent = ParentRegionQuery {
            key: Val(100),
            units: vec![Val(101)],
        };
        // `%4 = %2 + %3`, with `%3` the constant and `%1` the dominator of `%4`.
        let body = || {
            vec![
                extract(Val(10), (Val(1), Val(11)), 16, RegType::Lar),
                extract(Val(12), (Val(2), Val(13)), 16, RegType::Lar),
                constant(Val(3)),
                add(Val(2), Val(3), Val(4)),
                extract(Val(4), (Val(5), Val(6)), 16, RegType::Lar),
            ]
        };
        let map = |current: Vec<i64>| SsaMap {
            classes: vec![EquivalenceClass {
                expr_info: ExprInfo::default(),
                values: vec![Val(1), Val(4)],
            }],
            const_offsets: BTreeMap::from([
                (Val(1), vec![4]),
                (Val(2), vec![7]),
                (Val(4), current),
            ]),
            negated: BTreeMap::from([(Val(1), false), (Val(4), false)]),
        };

        // ⭐ SAME OFFSET, SAME SIGN — the add is redundant and every reader takes the dominator.
        let mut collapsing = map(vec![4]);
        let mut ops = body();
        let mut erased = Vec::new();
        let mut vals = Values::default();
        assert_eq!(
            collapsing.reconstruct_operation(
                0,
                1,
                0,
                &mut ops,
                &parent,
                &OutOfScopeUnitIndexMap,
                &mut vals,
                &mut erased,
            ),
            Reconstruction::Done
        );
        assert_eq!(collapsing.classes[0].values, vec![Val(1), Val(1)]);
        assert_eq!(erased, vec![Val(4)]);
        // ⛔ NOTHING WAS BUILT AND NOTHING WAS ERASED YET — e598 drains the list.
        assert_eq!(ops.len(), 5);
        assert_eq!(dialects::operands(&ops[4])[0], Val(1));

        // A difference of 5, agreed by every unit, so ONE immediate constant carries it.
        let mut rebuilding = map(vec![9]);
        let mut ops = body();
        let mut erased = Vec::new();
        let mut vals = Values::default();
        vals.mint();
        rebuilding.reconstruct_operation(
            0,
            1,
            0,
            &mut ops,
            &parent,
            &OutOfScopeUnitIndexMap,
            &mut vals,
            &mut erased,
        );
        // ⭐ BOTH NEW OPS LAND IN FRONT OF THE OLD ONE, which is where `OpBuilder builder(op)` puts
        // them, and the old op is left in the block for e598 to erase.
        assert_eq!(ops.len(), 7);
        let Op::Sentient(sentient::Op::ScalarConstant { value, result, .. }) = &ops[3] else {
            panic!("the rebuilt difference is one sentient.scalar_constant")
        };
        assert_eq!(*value, 5);
        let new_result = dialects::results(&ops[4])[0];
        assert!(matches!(
            &ops[4],
            Op::Sentient(sentient::Op::ScalarAdd { lhs, rhs, reg: None, .. })
                if *lhs == Val(1) && rhs == result
        ));
        assert_eq!(rebuilding.classes[0].values, vec![Val(1), new_result]);
        assert_eq!(rebuilding.const_offsets.get(&new_result), Some(&vec![9]));
        assert_eq!(dialects::operands(&ops[6])[0], new_result);

        // ⛔ A TRANSFER OP IS NEVER REBUILT: its address operand is repointed, or nothing happens.
        let mut transfers = SsaMap {
            classes: vec![EquivalenceClass {
                expr_info: ExprInfo::default(),
                values: vec![Val(1), Val(3)],
            }],
            const_offsets: BTreeMap::from([(Val(1), vec![8]), (Val(2), vec![8])]),
            negated: BTreeMap::from([(Val(1), false), (Val(3), false)]),
        };
        let mut ops = vec![
            extract(Val(10), (Val(1), Val(11)), 16, RegType::Lar),
            extract(Val(12), (Val(2), Val(13)), 16, RegType::Lar),
            extract(Val(2), (Val(3), Val(4)), 16, RegType::Lar),
        ];
        transfers.reconstruct_operation(
            0,
            1,
            0,
            &mut ops,
            &parent,
            &OutOfScopeUnitIndexMap,
            &mut vals,
            &mut Vec::new(),
        );
        assert_eq!(ops.len(), 3);
        assert!(matches!(
            &ops[2],
            Op::Sentient(sentient::Op::LoadAndExtractScalar { mutable_addr, .. })
                if *mutable_addr == Val(1)
        ));
    }

    /// e560 — the second value of a class has its address operand repointed at the dominator, and a
    /// value with no defining op in the unit — an iter arg, `isa<BlockArgument>` — is skipped.
    #[test]
    fn reduce_live_range_repoints_the_dominated_value_and_skips_a_block_argument() {
        let parent = ParentRegionQuery {
            key: Val(100),
            units: vec![Val(101)],
        };
        let mut map = SsaMap {
            classes: vec![EquivalenceClass {
                expr_info: ExprInfo::default(),
                values: vec![Val(1), Val(3), Val(99)],
            }],
            const_offsets: BTreeMap::from([(Val(1), vec![8]), (Val(2), vec![8])]),
            negated: BTreeMap::from([(Val(1), false), (Val(3), false)]),
        };
        let mut ops = vec![
            extract(Val(10), (Val(1), Val(11)), 16, RegType::Lar),
            extract(Val(12), (Val(2), Val(13)), 16, RegType::Lar),
            extract(Val(2), (Val(3), Val(4)), 16, RegType::Lar),
        ];
        let mut erased = Vec::new();
        let mut vals = Values::default();

        let failures = map.reduce_live_range(
            ElementSizeCheck::Required,
            &mut ops,
            &parent,
            &OutOfScopeUnitIndexMap,
            &mut vals,
            &mut erased,
        );

        assert!(failures.is_empty());
        assert!(matches!(
            &ops[2],
            Op::Sentient(sentient::Op::LoadAndExtractScalar { mutable_addr, .. })
                if *mutable_addr == Val(1)
        ));
    }

    /// e503 — the visit order IS the port: an op's results and a loop's arguments are filed in
    /// reverse, a loop's results forward from its yield, and a constant trip count is mapped only.
    #[test]
    fn map_all_values_files_the_results_in_reverse_and_the_loops_yield_forward() {
        let reg = Reg {
            locale: RegType::Lrf,
            index: None,
        };
        let ops = vec![
            constant(Val(1)),
            extract(Val(20), (Val(2), Val(3)), 16, RegType::Lar),
            Op::Sentient(sentient::Op::For {
                iv: Val(4),
                bound: Val(1),
                bound_reg: None,
                carried: vec![
                    sentient::Carried {
                        init: Val(2),
                        arg: Val(5),
                        result: Val(6),
                        reg,
                        program_header: false,
                        element_size: None,
                    },
                    sentient::Carried {
                        init: Val(3),
                        arg: Val(7),
                        result: Val(8),
                        reg,
                        program_header: false,
                        element_size: None,
                    },
                ],
                dbg_name: None,
                body: vec![Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(5), Val(7)],
                })],
            }),
        ];
        let mut canned = Canned {
            map: ExprInfoMap {
                exprs: vec![bucket(0, 1, vec![Val(9)])],
                buckets: vec![0],
            },
            flats: BTreeMap::from([(
                0,
                FlattenedExpr {
                    coeffs: vec![2, 5],
                    constraints: FlatAffineValueConstraints(3),
                    num_local_vars: 0,
                },
            )]),
        };

        let mut map = SsaMap::default();
        map.map_all_values(&mut canned, &ops);

        // The extract's data result before its address result; the second iter argument before the
        // first; then the loop's results, in order, from its yield.
        assert_eq!(
            map.classes[0].values,
            vec![Val(3), Val(2), Val(7), Val(5), Val(6), Val(8)]
        );
        // ⭐ THE TRIP COUNT IS MAPPED AS A CONSTANT, so it opens no class of its own.
        assert_eq!(map.const_offsets.get(&Val(1)), Some(&vec![5]));
        assert_eq!(map.classes.len(), 1);
    }

    /// e504 — a value computed into an `lrf` register and never read is handed out of the region;
    /// the `imm` constants beside it, which are read anyway, grow nothing.
    #[test]
    fn optimize_uniform_region_yielded_values_hands_out_the_unread_lrf_value() {
        let mut vals = Values::default();
        let unit_a = vals.mint();
        let unit_b = vals.mint();
        let const_a = vals.mint();
        let const_b = vals.mint();
        let offset = vals.mint();
        let mapping = vals.mint();
        let query = vals.mint();
        let sum = vals.mint();
        let r0_arg = vals.mint();

        let mut body = vec![
            get_unit(unit_a),
            get_unit(unit_b),
            Op::UniformRegions(UniformRegions::UniformizeRegions {
                regions: vec![LocalRegion {
                    arg: r0_arg,
                    units: vec![unit_a],
                    body: vec![
                        constant(const_a),
                        constant(const_b),
                        constant(offset),
                        Op::Uniform(uniform::Op::DefImmutableMapping {
                            result: mapping,
                            pairs: vec![(unit_a, const_a), (unit_b, const_b)],
                        }),
                        Op::Uniform(uniform::Op::QueryMap {
                            result: query,
                            map: mapping,
                            key: r0_arg,
                        }),
                        // `%sum = %query + %offset` in an `lrf` register, read by nobody.
                        Op::Sentient(sentient::Op::ScalarAdd {
                            lhs: query,
                            rhs: offset,
                            result: sum,
                            reg: Some(Reg {
                                locale: RegType::Lrf,
                                index: None,
                            }),
                            ty: ScalarTy::Index,
                            element_size: Some(Bits(16)),
                        }),
                        Op::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        }),
                    ],
                }],
                results: Vec::new(),
                yielded: Vec::new(),
            }),
        ];
        let parent = ParentRegionQuery {
            key: r0_arg,
            units: vec![unit_a, unit_b],
        };

        optimize_uniform_region_yielded_values(&mut body, &parent, &[], &mut vals);

        let Op::UniformRegions(UniformRegions::UniformizeRegions {
            regions,
            results,
            yielded,
        }) = &body[2]
        else {
            panic!("the op keeps its slot")
        };
        // ⛔ EXACTLY ONE NEW RESULT: the three `imm` constants are read by the mapping and the add,
        // and an `imm` locale is not one a region hands a value out through anyway.
        assert_eq!(results.len(), 1);
        assert_eq!(
            yielded,
            &vec![YieldedReg {
                element_size: Some(Bits(16)),
                locale: RegType::Lrf,
            }]
        );
        // ⭐ THE ONE-REGION OP BECAME A TWO-REGION ONE, region 1 covering the unit region 0 leaves.
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[1].units, vec![unit_b]);
        let handed_out = dialects::results(&regions[0].body[5])[0];
        assert!(matches!(
            regions[0].body.last(),
            Some(Op::Uniform(uniform::Op::Yield { operands })) if *operands == vec![handed_out]
        ));
    }

    /// A model and a rung, so the program is typed; nothing here reads either.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// A `PropagationAnalysis` THAT FAILED, and whose every expression is irresolvable — so nothing is
    /// classified and e598's reduction has nothing to move.
    struct Failed;

    impl PropagationAnalysis for Failed {
        type Units = OutOfScopeUnitIndexMap;

        fn affine_expression(&mut self, _val: Val) -> ExprInfoMap {
            ExprInfoMap {
                exprs: vec![Some(PropagatedExpr {
                    propagated_map: PropagatedMap { id: 0, num_dims: 1 },
                    propagated_args: Vec::new(),
                    cannot_be_resolved: true,
                })],
                buckets: vec![0],
            }
        }

        fn flattened_affine_expr(&self, _map: PropagatedMap) -> FlattenedExpr {
            todo!("unreachable: every expression this double answers is irresolvable")
        }

        fn unit_index_map(&self) -> Self::Units {
            OutOfScopeUnitIndexMap
        }

        fn is_propagation_successful(&self) -> bool {
            false
        }
    }

    /// e598 — the pass entry over a two-unit program: BOTH units' one-region
    /// `uniform.uniformize_regions` hands its unread `lrf` value out, and ⛔ THE FAILED PROPAGATION IS
    /// REPORTED WITHOUT STOPPING THE WALK, which is what the reference's missing `return` means.
    #[test]
    fn e598_walks_every_unit_and_reports_the_failed_propagation_as_data() {
        let mut vals = Values::default();
        let unit_a = vals.mint();
        let unit_b = vals.mint();
        let key = vals.mint();
        let unit_body = |vals: &mut Values| {
            let (const_a, const_b, offset) = (vals.mint(), vals.mint(), vals.mint());
            let (mapping, query, sum, r0_arg) =
                (vals.mint(), vals.mint(), vals.mint(), vals.mint());
            vec![
                get_unit(unit_a),
                get_unit(unit_b),
                Op::UniformRegions(UniformRegions::UniformizeRegions {
                    regions: vec![LocalRegion {
                        arg: r0_arg,
                        units: vec![unit_a],
                        body: vec![
                            constant(const_a),
                            constant(const_b),
                            constant(offset),
                            Op::Uniform(uniform::Op::DefImmutableMapping {
                                result: mapping,
                                pairs: vec![(unit_a, const_a), (unit_b, const_b)],
                            }),
                            Op::Uniform(uniform::Op::QueryMap {
                                result: query,
                                map: mapping,
                                key: r0_arg,
                            }),
                            Op::Sentient(sentient::Op::ScalarAdd {
                                lhs: query,
                                rhs: offset,
                                result: sum,
                                reg: Some(Reg {
                                    locale: RegType::Lrf,
                                    index: None,
                                }),
                                ty: ScalarTy::Index,
                                element_size: Some(Bits(16)),
                            }),
                            Op::Uniform(uniform::Op::Yield {
                                operands: Vec::new(),
                            }),
                        ],
                    }],
                    results: Vec::new(),
                    yielded: Vec::new(),
                }),
            ]
        };
        let unit_of = |body: Vec<Op>| ProgramUnit {
            on: Units::of(
                DfirUnit::Lxlu,
                &[(DfirUnit::Lxlu, unit_a), (DfirUnit::Lxlu, unit_b)],
            )
            .expect("two units of one kind"),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        let first = unit_of(unit_body(&mut vals));
        let second = unit_of(unit_body(&mut vals));
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(first, vec![second]),
            bound: core::marker::PhantomData,
        };

        let (propagation, failures) =
            run_on_operation(&mut program, &mut Failed, &[key, key], &mut vals);

        assert_eq!(propagation, Some(PropagationFailure));
        assert!(failures.is_empty());
        for unit in program.units.iter() {
            let Op::UniformRegions(UniformRegions::UniformizeRegions {
                regions, results, ..
            }) = &unit.body[2]
            else {
                panic!("the op keeps its slot in every unit")
            };
            assert_eq!(results.len(), 1);
            assert_eq!(regions.len(), 2);
            assert_eq!(regions[1].units, vec![unit_b]);
        }
    }
}
