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

//! `ScalarSimplifications.cpp` — 5 of the campaign's 656 units (dependency level(s) [1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e372_areAllExprsValidToTransform` | 372 | 1 | 50 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:716` |
//! | `e373_areFlatExprsAndArgsIdentical` | 373 | 1 | 29 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:840` |
//! | `e472_createNewOpOrMap` | 472 | 2 | 63 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:769` |
//! | `e534_lightWeightSimplifyBinaryArithmetic` | 534 | 3 | 183 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:504` |
//! | `e580_runOldLightWeightSimplifications` | 580 | 4 | 25 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:690` |

// ⛔ THE TWO PORTED PREDICATES ARE REACHED ONLY FROM THIS FILE'S TESTS until `e472_createNewOpOrMap`
// and `e534_lightWeightSimplifyBinaryArithmetic` (levels 2/3) land and call them. CI runs clippy with
// `-D warnings`, so without this the first ported leaf fails the gate.
// ⭐ REMOVE THIS WITH e534: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::sentient::dialects::{Definitions, Val};
use crate::transform::sentient::analyses::{ExprInfoMap, UnitIndex};

/// `FlatExprType` (`ScalarSimplifications.cpp:35`) FOR ONE UNIT — every flattened affine expression
/// of that unit's propagated map, each a coefficient per dim then per local var then the constant.
///
/// ⛔ ONE UNIT, NOT ONE EXPRESSION: `std::vector<FlatExprType> flat_exprs` is indexed by the unit's
/// POSITION in `indices` (`:339-359`), and both predicates below then reject any unit whose list holds
/// more than the one expression `getFlattenedAffineExprs` produced.
pub type FlatExpr = Vec<Vec<i64>>;

/// `dcc::utils::isInductionVariable` (`Analyses/Utils.cpp:128`) — a `sentient.for`'s own induction
/// variable, and nothing else.
///
/// ⭐ STRUCTURAL, SO NOT AN OUT-OF-SCOPE `todo!`: [`Definitions::for_arg_of`] performs the reference's
/// `cast<BlockArgument>(val).getOwner()->getParentOp()` walk, and its position 0 IS `getInductionVar()`
/// — so a carried iter arg, an op result and a region argument of anything else all answer `false`.
fn is_induction_variable(val: Val, defs: Definitions<'_>) -> bool {
    matches!(defs.for_arg_of(val), Some((_, 0)))
}

/// `result_info->getExprInfoAt(indices[unit_idx])->propagated_args_`.
///
/// ⛔⛔ THE TWO INDEXES ARE NOT THE SAME NUMBER: `flat_exprs` is read by POSITION in the walk while
/// `result_info` is read by `indices[position]`, the unit's index in
/// [`UnitIndexMap`](crate::transform::sentient::analyses::UnitIndexMap). Reading `result_info` by the
/// position would silently take another unit's arguments.
fn propagated_args<'a>(
    result_info: &'a ExprInfoMap,
    indices: &[UnitIndex],
    unit_idx: usize,
) -> Option<&'a [Val]> {
    let expr_info = result_info.expr_info_at(*indices.get(unit_idx)?)?;
    Some(&expr_info.propagated_args)
}

/// Replaces: e372_areAllExprsValidToTransform
///
/// Whether every selected unit's flattened expression is one the uniformizing rewrite can build: one
/// expression, the same dim count throughout, unit coefficients, and no loop iterator among its args.
///
/// ⛔ A MISSING `ExprInfo` REJECTS RATHER THAN ABORTS. `simplifyBinaryOperation` builds `flat_exprs`
/// out of these very buckets behind `DT_CHECK_MSG(expr_info, "Expecting valid ExprInfo for unit")`
/// (`:344`), so this is unreachable — and refusing the transform is the conservative answer.
///
/// ⭐ `expression.size() > 3 || < 1` IS AN `else if` CHAIN, so a size of 1 (a bare constant) passes
/// every test and a `is_load_store` expression is already capped at 2.
#[must_use]
pub fn are_all_exprs_valid_to_transform(
    flat_exprs: &[FlatExpr],
    indices: &[UnitIndex],
    result_info: &ExprInfoMap,
    is_load_store: bool,
    defs: Definitions<'_>,
) -> bool {
    // `unsigned dim_size = flat_exprs[0][0].size()` — vacuous when no unit was selected.
    let Some(dim_size) = flat_exprs.first().and_then(|first| first.first()).map(Vec::len) else {
        return true;
    };
    for (unit_idx, flat_expr_at_idx) in flat_exprs.iter().enumerate() {
        if flat_expr_at_idx.len() > 1 {
            return false;
        }
        let Some(expression) = flat_expr_at_idx.first() else {
            return false;
        };
        if is_load_store && expression.len() > 2 {
            return false;
        }
        if dim_size != expression.len() {
            return false;
        }
        if expression.len() > 3 || expression.is_empty() {
            return false;
        } else if expression.len() == 2 {
            let Some(args) = propagated_args(result_info, indices, unit_idx) else {
                return false;
            };
            if args.first().is_some_and(|arg| is_induction_variable(*arg, defs))
                || expression[0] != 1
            {
                return false;
            }
            if is_load_store && expression[1] != 0 {
                return false;
            }
        } else if expression.len() == 3 {
            let Some(args) = propagated_args(result_info, indices, unit_idx) else {
                return false;
            };
            if args.iter().any(|arg| is_induction_variable(*arg, defs))
                || expression[2] != 0
                || expression[0] != 1
                || expression[1] != 1
            {
                return false;
            }
        }
    }
    true
}

/// Replaces: e373_areFlatExprsAndArgsIdentical
///
/// Whether every selected unit agrees with the first on dim count, on the `dim_idx` coefficient and on
/// the propagated arguments — the test that one op can stand for all of them.
///
/// ⛔ `DT_CHECK(num_flat_exprs > 1)` (`:844`) IS VACUOUS HERE RATHER THAN AN ABORT: with one
/// expression there is nothing to disagree with, which is the answer the loop already gives.
///
/// ⛔ THE ARGUMENT CHECKS ARE GATED ON THE SIZE, NOT ON THE ARGUMENT LIST: `>= 2` reads arg 0 and
/// `== 3` reads arg 1, because a flattened expression is one coefficient per dim plus the constant
/// and the caller has already refused any local variables (`:355-357`).
#[must_use]
pub fn are_flat_exprs_and_args_identical(
    flat_exprs: &[FlatExpr],
    indices: &[UnitIndex],
    result_info: &ExprInfoMap,
    dim_idx: usize,
) -> bool {
    let Some(first_flat_expr) = flat_exprs.first().and_then(|first| first.first()) else {
        return true;
    };
    let first_args = propagated_args(result_info, indices, 0);
    for (unit_idx, flat_expr_at_idx) in flat_exprs.iter().enumerate().skip(1) {
        let Some(flat_expr_at_idx) = flat_expr_at_idx.first() else {
            return false;
        };
        if flat_expr_at_idx.len() != first_flat_expr.len() {
            return false;
        }
        if flat_expr_at_idx.get(dim_idx) != first_flat_expr.get(dim_idx) {
            return false;
        }
        let (Some(args), Some(first_args)) =
            (propagated_args(result_info, indices, unit_idx), first_args)
        else {
            return false;
        };
        if flat_expr_at_idx.len() >= 2 && args.first() != first_args.first() {
            return false;
        }
        if flat_expr_at_idx.len() == 3 && args.get(1) != first_args.get(1) {
            return false;
        }
    }
    true
}

// crustify:todo: e472_createNewOpOrMap
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:769  (63 body lines, level 2)
//   original  : Value sentient::createNewOpOrMap(OpBuilder& builder, std::vector<FlatExprType>& flat_exprs, SmallVector<unsigned>& indices, PropagationAnalysis& expr_prop_analysis, SmallVector<Value>& units, Operation* op, Type operand_type, Value query_key, PropagationAnalysis::ExprInfoMap* result_info, unsigned o
//   calls     : e252_size, e373_areFlatExprsAndArgsIdentical

// crustify:todo: e534_lightWeightSimplifyBinaryArithmetic
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:504  (183 body lines, level 3)
//   original  : void sentient::lightWeightSimplifyBinaryArithmetic( OpBuilder& builder, PropagationAnalysis& expr_prop_analysis, Operation* op, std::vector<mlir::Operation*>& tobe_deleted)
//   calls     : e240_selectIndicesForUnits, e252_size, e392_getQueryKeyAndUnitsFromParentRegion, e472_createNewOpOrMap

// crustify:todo: e580_runOldLightWeightSimplifications
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:690  (25 body lines, level 4)
//   original  : LogicalResult sentient::runOldLightWeightSimplifications(Operation* op)
//   calls     : e534_lightWeightSimplifyBinaryArithmetic

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::sentient::dialects::{Op, sentient};
    use crate::transform::sentient::analyses::PropagatedExpr;

    /// `sentient.for %iv = %bound { }` — the loop whose induction variable both predicates see.
    fn loop_over(iv: Val, bound: Val) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            carried: Vec::new(),
            dbg_name: None,
            body: Vec::new(),
        })
    }

    /// An `ExprInfoMap` of one bucket per unit, in unit order — `getListIdxFromUnitIdx` is the identity.
    fn expr_info_map(args_per_unit: &[&[Val]]) -> ExprInfoMap {
        ExprInfoMap {
            exprs: args_per_unit
                .iter()
                .map(|args| {
                    Some(PropagatedExpr {
                        propagated_args: args.to_vec(),
                        ..PropagatedExpr::default()
                    })
                })
                .collect(),
            buckets: (0..args_per_unit.len()).collect(),
        }
    }

    /// e372 — `1 * %arg + 0` on both units transforms; an induction variable argument, a coefficient
    /// other than 1 and a non-zero constant under `is_load_store` each refuse. The last assertion is
    /// the index trap: swapping `indices` makes position 1 read the bucket holding the iterator.
    #[test]
    fn every_unit_needs_a_unit_coefficient_expression_over_a_non_iterator() {
        let scope = vec![loop_over(Val(10), Val(11))];
        let regions: [&[Op]; 1] = [scope.as_slice()];
        let defs = Definitions::from_innermost(&regions);
        let indices = [UnitIndex(0), UnitIndex(1)];
        let identity: Vec<FlatExpr> = vec![vec![vec![1, 0]], vec![vec![1, 0]]];

        let info = expr_info_map(&[&[Val(2)], &[Val(3)]]);
        assert!(are_all_exprs_valid_to_transform(
            &identity, &indices, &info, true, defs
        ));

        let with_iterator = expr_info_map(&[&[Val(10)], &[Val(3)]]);
        assert!(!are_all_exprs_valid_to_transform(
            &identity,
            &indices,
            &with_iterator,
            true,
            defs
        ));

        let scaled: Vec<FlatExpr> = vec![vec![vec![2, 0]], vec![vec![2, 0]]];
        assert!(!are_all_exprs_valid_to_transform(
            &scaled, &indices, &info, true, defs
        ));

        let offset: Vec<FlatExpr> = vec![vec![vec![1, 4]], vec![vec![1, 4]]];
        assert!(!are_all_exprs_valid_to_transform(
            &offset, &indices, &info, true, defs
        ));
        assert!(
            are_all_exprs_valid_to_transform(&offset, &indices, &info, false, defs),
            "the non-zero constant is refused only for a load or a store"
        );

        let swapped = [UnitIndex(1), UnitIndex(0)];
        assert!(
            !are_all_exprs_valid_to_transform(&identity, &swapped, &with_iterator, true, defs),
            "position 1 reads `indices[1]`, which is bucket 0 and holds the iterator"
        );
    }

    /// e373 — two units agree when the dim count, the `dim_idx` coefficient and the arguments all
    /// match; a differing argument and a differing coefficient each refuse.
    #[test]
    fn identical_means_the_same_dim_count_coefficient_and_arguments() {
        let indices = [UnitIndex(0), UnitIndex(1)];
        let identity: Vec<FlatExpr> = vec![vec![vec![1, 0]], vec![vec![1, 0]]];

        let same = expr_info_map(&[&[Val(2)], &[Val(2)]]);
        assert!(are_flat_exprs_and_args_identical(
            &identity, &indices, &same, 0
        ));

        let differing = expr_info_map(&[&[Val(2)], &[Val(3)]]);
        assert!(!are_flat_exprs_and_args_identical(
            &identity, &indices, &differing, 0
        ));

        let scaled: Vec<FlatExpr> = vec![vec![vec![1, 0]], vec![vec![2, 0]]];
        assert!(!are_flat_exprs_and_args_identical(
            &scaled, &indices, &same, 0
        ));
        assert!(
            are_flat_exprs_and_args_identical(&scaled, &indices, &same, 1),
            "only the `dim_idx` coefficient is compared"
        );
    }
}
