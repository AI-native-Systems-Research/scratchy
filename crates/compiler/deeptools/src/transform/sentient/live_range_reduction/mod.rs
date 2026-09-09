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

use crate::islands::sentient::dialects::{
    Definitions, Op, Val, element_size, sentient, symbol, uniform, value_reg_locale,
};
use crate::islands::sentient::print;

/// `affine::FlatAffineValueConstraints` — THE LOCAL-VARIABLE CONSTRAINT SYSTEM OF ONE FLATTENED
/// AFFINE EXPRESSION, opaque here.
///
/// ⛔ MLIR UPSTREAM, AND NOTHING IN THIS CAMPAIGN BUILDS OR READS ONE. The only producer is
/// `mlir::getFlattenedAffineExpr` (`LiveRangeReduction.cpp:305`, out of campaign scope) and the only
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
    /// Overwrites all three fields — the one way `getBaseExpr` (`:337`, `:349`) fills an `ExprInfo`.
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
    /// ⭐ EVERY COEFFICIENT IS FOLLOWED BY `", "`, INCLUDING THE LAST (`:145`), and "constrains" is
    /// the reference's own spelling (`:154`) — this is its output text, not prose.
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
/// the reference `emitOpError("can't find global ancestor")` and fail the pass (`:1147-1149`).
///
/// ⛔ AND THE SRC/DST SPLIT IS THE REFERENCE'S INTENT, NOT ITS CODE. `val ==
/// load_store_op.getSrcMutableAddr()` (`:1123`) compares an op RESULT against that op's own operand
/// 2 and so is always false, leaving the `src` branch dead and every `load_and_store` address
/// walking through `dst`; the `.td` names the two results `src_res`/`dst_res` as "the final,
/// potentially updated, address" of each side (`SentientOps.td:760-763`), so this dispatches on WHICH
/// RESULT `val` is.
#[must_use]
pub fn get_first_global_or_constant_ancestor(
    val: Val,
    uniform_region: &[Op],
    defs: Definitions<'_>,
) -> Option<Val> {
    let mut val = val;
    loop {
        // ⭐ `DT_CHECK(isa<sentient::ForOp>(op))` (`:1113`) IS A FACT ABOUT THE ISLAND, not a check:
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
                    // ⭐ FOLLOW THE OPERAND THAT IS NOT A CONSTANT, PREFERRING `lhs` (`:1136-1140`)
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
                    // test, which is why a terminal op outside the region still answers (`:1145`).
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

// crustify:todo: e304_printSsaMap
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:218  (18 body lines, level 1)
//   original  : void LiveRangeReductionPass::printSsaMap(raw_ostream& os)
//   calls     : e057_print, e252_size

// crustify:todo: e305_getBaseExpr
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:271  (116 body lines, level 1)
//   original  : LogicalResult LiveRangeReductionPass::getBaseExpr( PropagationAnalysis& expr_prop_analysis, const Value value, ExprInfo& expr_info, std::vector<int64_t>& const_val, bool& negated)
//   calls     : e056_set, e252_size

// crustify:todo: e306_getRootIterArg
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:393  (72 body lines, level 1)
//   original  : Value LiveRangeReductionPass::getRootIterArg(Value val)
//   calls     : e252_size

// crustify:todo: e307_cloneValueToRegion
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:917  (40 body lines, level 1)
//   original  : std::optional<Value> cloneValueToRegion(Value val, Region& region, OpBuilder& builder)
//   calls     : e252_size

// crustify:todo: e440_addToMap
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:241  (25 body lines, level 2)
//   original  : void LiveRangeReductionPass::addToMap(PropagationAnalysis& expr_prop_analysis, Value& value)
//   calls     : e252_size, e305_getBaseExpr

// crustify:todo: e441_findDominantValue
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:480  (38 body lines, level 2)
//   original  : LogicalResult LiveRangeReductionPass::findDominantValue( SmallVector<Value> ssa_list, const int current_value_index, int& dominant_value_index, const DominanceInfo& dominance_info)
//   calls     : e058_areElementSizeIdentical, e060_areReglocalesMatching, e306_getRootIterArg

// crustify:todo: e442_addResultToYield
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:961  (140 body lines, level 2)
//   original  : uniform::UniformizeRegionsOp LiveRangeReductionPass::addResultToYield( uniform::UniformizeRegionsOp uniformize_op, SmallVector<Operation*>& ops_to_be_delected, int region_num, int op_num, int result_idx)
//   calls     : e059_getFirstGlobalOrConstantAncestor, e239_getOperationOfBlock, e252_size, e307_cloneValueToRegion, e392_getQueryKeyAndUnitsFromParentRegion

// crustify:todo: e443_createMapAndQuery
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1216  (43 body lines, level 2)
//   original  : Operation* LiveRangeReductionPass::createMapAndQuery( OpBuilder& builder, std::vector<int64_t>& new_const_val, Operation* op)
//   calls     : e252_size, e392_getQueryKeyAndUnitsFromParentRegion

// crustify:todo: e502_reconstructOperation
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:546  (259 body lines, level 3)
//   original  : LogicalResult LiveRangeReductionPass::reconstructOperation( SmallVector<Value>& ssa_list, int current_value_index, int dominant_value_index)
//   calls     : e057_print, e252_size, e443_createMapAndQuery

// crustify:todo: e503_mapAllValues
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:827  (48 body lines, level 3)
//   original  : void LiveRangeReductionPass::mapAllValues(PropagationAnalysis& expr_prop, Operation* unit_op)
//   calls     : e440_addToMap

// crustify:todo: e504_optimizeUniformRegionYieldedValues
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1167  (47 body lines, level 3)
//   original  : void LiveRangeReductionPass::optimizeUniformRegionYieldedValues( ModuleOp module_op)
//   calls     : e239_getOperationOfBlock, e252_size, e442_addResultToYield

// crustify:todo: e560_reduceLiveRange
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:885  (28 body lines, level 4)
//   original  : void LiveRangeReductionPass::reduceLiveRange(PropagationAnalysis& expr_prop, Operation* unit_op)
//   calls     : e252_size, e441_findDominantValue, e502_reconstructOperation

// crustify:todo: e598_runOnOperation
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1267  (45 body lines, level 5)
//   original  : void LiveRangeReductionPass::runOnOperation()
//   calls     : e304_printSsaMap, e503_mapAllValues, e504_optimizeUniformRegionYieldedValues, e560_reduceLiveRange


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
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
}
