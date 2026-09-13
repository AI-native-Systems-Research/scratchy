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

#![allow(dead_code)]
// ⛔ NOTHING IN THIS FILE IS WIRED INTO THE PIPELINE YET — `e580_runOldLightWeightSimplifications`
// (level 4) is what reaches the item below, through e534. CI runs clippy with `-D warnings`, so
// without this the first ported leaf here fails the gate. ⭐ REMOVE THIS WITH e580.

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Op, Val, sentient as ops, uniform};
use crate::transform::sentient::analyses::{ExprInfoMap, PropagationAnalysis};

/// `FlatExprType` (`:35`) — ONE UNIT'S FLATTENED AFFINE EXPRESSIONS, coefficients then constant.
///
/// ⛔ EVERY READER OF ONE IN THIS FILE READS `[0]`: the outer list is per result of the propagated
/// affine map and the simplifications only ever transform a single-result one.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct FlatExpr(pub(crate) Vec<Vec<i64>>);

impl FlatExpr {
    /// `flat_expr[0]` — ⭐ EMPTY WHERE THE REFERENCE DEREFERENCES `std::vector::operator[]` past the
    /// end, which every caller's `size()` chain then declines.
    #[must_use]
    fn first(&self) -> &[i64] {
        self.0.first().map_or(&[], Vec::as_slice)
    }
}


// crustify:todo: e372_areAllExprsValidToTransform
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:716  (50 body lines, level 1)
//   original  : bool sentient::areAllExprsValidToTransform( std::vector<FlatExprType>& flat_exprs, SmallVector<unsigned>& indices, PropagationAnalysis::ExprInfoMap* result_info, bool is_load_store)
//   calls     : e252_size

// crustify:todo: e373_areFlatExprsAndArgsIdentical
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:840  (29 body lines, level 1)
//   original  : bool sentient::areFlatExprsAndArgsIdentical( std::vector<FlatExprType>& flat_exprs, SmallVector<unsigned>& indices, PropagationAnalysis::ExprInfoMap* result_info, unsigned dim_idx)
//   calls     : e252_size

/// `sentient::areFlatExprsAndArgsIdentical` — e373, level 1, not yet ported.
///
/// ⛔ IT IS e373 AND IT IS NOT PORTED YET: whether every unit's flattened expression AND every
/// unit's propagated arguments agree, which is what lets one value stand for all of them.
fn are_flat_exprs_and_args_identical(
    flat_exprs: &[FlatExpr],
    indices: &[usize],
    result_info: ExprInfoMap,
    dim_idx: usize,
) -> bool {
    let _ = (flat_exprs, indices, result_info, dim_idx);
    todo!(
        "areFlatExprsAndArgsIdentical (senpass e373, ScalarSimplifications.cpp:840) is not ported \
         yet — the per-unit comparison of both the flattened expression and the propagated args"
    )
}

/// Replaces: e472_createNewOpOrMap
///
/// The one value that stands for a simplified operand: a `sentient.scalar_constant`, a propagated
/// argument as it is, or — when the units disagree — a `uniform.def_immutable_mapping` of one value
/// per unit read back through a `uniform.query_map`.
///
/// ⛔ `first_flat_expr_size` DECIDES THE SHAPE FOR EVERY UNIT, including in the per-unit loop
/// (`:809-826`): unit 0's expression length is the whole discriminator.
/// ⛔ SIZE 2 IS THE ONE ASYMMETRIC CASE — only `operand_idx == 1` becomes a constant, the other index
/// answers with the propagated argument (`:788-793`).
/// ⛔ A SIZE THAT IS NOT 1, 2 OR 3 FALLS THROUGH TO THE MAPPING PATH, whose own chain then pushes
/// NOTHING and builds a mapping of no values. The reference has no `else` on either chain.
/// ⭐ THE SAVE/RESTORE OF THE INSERTION POINT IS DROPPABLE MECHANISM; what is not is that every op
/// created here lands immediately BEFORE `at`, in creation order.
pub(crate) fn create_new_op_or_map(
    scope: &mut Vec<Op>,
    at: usize,
    flat_exprs: &[FlatExpr],
    indices: &[usize],
    units: &[Val],
    result_info: ExprInfoMap,
    operand_idx: usize,
    ty: ScalarTy,
    query_key: Val,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    values: &mut Values,
) -> Val {
    let first_flat_expr = flat_exprs.first().map(FlatExpr::first).unwrap_or(&[]);
    let first_flat_expr_size = first_flat_expr.len();
    if flat_exprs.len() == 1
        || are_flat_exprs_and_args_identical(flat_exprs, indices, result_info, operand_idx)
    {
        match first_flat_expr_size {
            1 => return scalar_constant(scope, at, first_flat_expr[0], ty, values),
            2 if operand_idx == 1 => {
                return scalar_constant(scope, at, first_flat_expr[operand_idx], ty, values);
            }
            // `getFirstExprInfo()->propagated_args_[operand_idx]` — unit index 0.
            2 | 3 => {
                if let Some(arg) = propagated_arg(expr_prop_analysis, result_info, 0, operand_idx) {
                    return arg;
                }
            }
            _ => {}
        }
    }
    if units.len() != flat_exprs.len() {
        todo!(
            "createNewOpOrMap: DT_CHECK_MSG(units.size() == num_flat_exprs, \"Expecting a flattened \
             expression for each unit\") (ScalarSimplifications.cpp:803) — {} units against {} \
             flattened expressions",
            units.len(),
            flat_exprs.len()
        )
    }
    let mut mapped: Vec<Val> = Vec::new();
    let mut inserted = 0usize;
    for (unit_idx, cur_flat_expr) in flat_exprs.iter().enumerate() {
        let expr_info_idx = indices.get(unit_idx).copied().unwrap_or(0);
        let cur = cur_flat_expr.first();
        let constant_at = |index: usize| cur.get(index).copied().unwrap_or(0);
        match first_flat_expr_size {
            1 => {
                mapped.push(scalar_constant(scope, at + inserted, constant_at(0), ty, values));
                inserted += 1;
            }
            2 if operand_idx == 1 => {
                let value = constant_at(operand_idx);
                mapped.push(scalar_constant(scope, at + inserted, value, ty, values));
                inserted += 1;
            }
            2 | 3 => {
                if let Some(arg) =
                    propagated_arg(expr_prop_analysis, result_info, expr_info_idx, operand_idx)
                {
                    mapped.push(arg);
                }
            }
            _ => {}
        }
    }
    let map = values.mint();
    let query = values.mint();
    scope.insert(
        at + inserted,
        Op::Uniform(uniform::Op::DefImmutableMapping {
            result: map,
            pairs: units.iter().copied().zip(mapped).collect(),
        }),
    );
    scope.insert(
        at + inserted + 1,
        Op::Uniform(uniform::Op::QueryMap {
            result: query,
            map,
            key: query_key,
        }),
    );
    query
}

/// `result_info->getExprInfoAt(at)->propagated_args_[operand_idx]` — the one read of the analysis this
/// unit makes, and `None` where the reference indexes a `SmallVector` past its end.
fn propagated_arg(
    expr_prop_analysis: &mut impl PropagationAnalysis,
    result_info: ExprInfoMap,
    at: usize,
    operand_idx: usize,
) -> Option<Val> {
    expr_prop_analysis
        .propagated_args(result_info, at)
        .get(operand_idx)
        .copied()
}

/// `sentient::ConstantOp::create(builder, loc, operand_type, value)` inserted before `at`.
fn scalar_constant(
    scope: &mut Vec<Op>,
    at: usize,
    value: i64,
    ty: ScalarTy,
    values: &mut Values,
) -> Val {
    let result = values.mint();
    scope.insert(
        at,
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: ops::RegType::Imm,
            ty,
            is_symbol: false,
        }),
    );
    result
}

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
    use super::{FlatExpr, create_new_op_or_map};
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Op, Val, sentient as ops};
    use crate::transform::sentient::analyses::{ExprInfoMap, PropagationAnalysis};

    /// AN ANALYSIS THAT ANSWERS WHAT THE TEST SAYS — `PropagationAnalysis` is out of campaign scope.
    struct StatedAnalysis;

    impl PropagationAnalysis for StatedAnalysis {
        fn are_expressions_same(&mut self, _val1: Val, _val2: Val) -> bool {
            todo!("e472 never asks whether two expressions are the same")
        }

        fn propagated_args(&mut self, _map: ExprInfoMap, at: usize) -> Vec<Val> {
            vec![Val(50 + at as u32), Val(60 + at as u32)]
        }
    }

    /// A minter that has already issued `issued` values, so a fixture's own [`Val`]s cannot collide.
    fn values_after(issued: u32) -> Values {
        let mut values = Values::default();
        for _ in 0..issued {
            values.mint();
        }
        values
    }

    /// e472 — the non-uniformization case: one unit's flattened expression of length 1 becomes a
    /// `sentient.scalar_constant` before the op, and one of length 3 answers with the propagated
    /// argument itself, creating nothing.
    #[test]
    fn a_single_unit_expression_becomes_a_constant_or_the_propagated_argument() {
        let mut scope = vec![Op::Sentient(ops::Op::Nop { dbg_name: None })];
        let mut values = values_after(10);
        assert_eq!(
            create_new_op_or_map(
                &mut scope,
                0,
                &[FlatExpr(vec![vec![7]])],
                &[0],
                &[Val(80)],
                ExprInfoMap(0),
                0,
                ScalarTy::Index,
                Val(81),
                &mut StatedAnalysis,
                &mut values,
            ),
            Val(10)
        );
        assert_eq!(
            scope[0],
            Op::Sentient(ops::Op::ScalarConstant {
                value: 7,
                result: Val(10),
                reg_locale: ops::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            })
        );
        assert_eq!(
            create_new_op_or_map(
                &mut scope,
                1,
                &[FlatExpr(vec![vec![1, 1, 0]])],
                &[0],
                &[Val(80)],
                ExprInfoMap(0),
                1,
                ScalarTy::Index,
                Val(81),
                &mut StatedAnalysis,
                &mut values,
            ),
            Val(60)
        );
        assert_eq!(scope.len(), 2);
    }

    /// e472 — more than one unit asks whether every unit agrees, which is e373 and not ported.
    #[test]
    #[should_panic(expected = "senpass e373")]
    fn several_units_reach_the_unported_identical_expression_test() {
        let mut scope = vec![Op::Sentient(ops::Op::Nop { dbg_name: None })];
        create_new_op_or_map(
            &mut scope,
            0,
            &[FlatExpr(vec![vec![7]]), FlatExpr(vec![vec![9]])],
            &[0, 1],
            &[Val(80), Val(82)],
            ExprInfoMap(0),
            0,
            ScalarTy::Index,
            Val(81),
            &mut StatedAnalysis,
            &mut values_after(10),
        );
    }
}
