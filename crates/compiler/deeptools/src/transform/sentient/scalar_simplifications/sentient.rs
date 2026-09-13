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
// ⛔ NOTHING IN THIS FILE IS WIRED INTO THE PIPELINE YET — the pass entry that reaches
// `e580_runOldLightWeightSimplifications` is e597 (level 5), behind an `OldLightweightSimplification`
// flag that ships `cl::init(false)`. CI runs clippy with `-D warnings`, so without this the leaves
// here fail the gate. ⭐ REMOVE THIS WITH THAT DRIVER, not with an anchor.

use super::{ConstSink, QueryKeyAndUnits};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, erase_defining_op, operands, regions_mut, replace_all_uses_with,
    sentient as ops, set_operand, uniform, use_count,
};
use crate::transform::sentient::analyses::{ExprInfoMap, PropagationAnalysis, UnitIndexMap};
use crate::transform::sentient::register_packing::constant_target_values;
use crate::transform::sentient::utils::{ConstKind, is_constant, select_indices_for_units};

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
    pub(super) fn first(&self) -> &[i64] {
        self.0.first().map_or(&[], Vec::as_slice)
    }
}


// crustify:todo: e372_areAllExprsValidToTransform
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:716  (50 body lines, level 1)
//   original  : bool sentient::areAllExprsValidToTransform( std::vector<FlatExprType>& flat_exprs, SmallVector<unsigned>& indices, PropagationAnalysis::ExprInfoMap* result_info, bool is_load_store)
//   calls     : e252_size

/// WHICH CALLER `areAllExprsValidToTransform` ANSWERS FOR — its defaulted `bool is_load_store`
/// (`:42`), whose effect is one further length limit (`:729`) and one further coefficient test
/// (`:748`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExprUse {
    /// `is_load_store == false` — e532 and e534.
    BinaryOperation,
    /// `is_load_store == true` — e533's operand lambda.
    LoadStore,
}

/// `sentient::areAllExprsValidToTransform` — e372, level 1, not yet ported.
///
/// ⛔ IT IS e372 AND IT IS NOT PORTED YET: whether every unit's flattened expression is single,
/// between 1 and 3 terms long, the same length as unit 0's, coefficients all 1, no constant on the
/// 3-term case — and written over no induction variable.
pub(super) fn are_all_exprs_valid_to_transform(
    flat_exprs: &[FlatExpr],
    indices: &[usize],
    result_info: ExprInfoMap,
    use_of: ExprUse,
) -> bool {
    let _ = (flat_exprs, indices, result_info, use_of);
    todo!(
        "areAllExprsValidToTransform (senpass e372, ScalarSimplifications.cpp:716) is not ported \
         yet — the per-unit agreement of dimension size, coefficients and induction-variable freedom"
    )
}

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
/// answers with the propagated argument (`:791-798`).
/// ⛔ A SIZE THAT IS NOT 1, 2 OR 3 FALLS THROUGH TO THE MAPPING PATH, whose own chain then pushes
/// NOTHING and builds a mapping of no values. The reference has no `else` on either chain.
/// ⛔ THE TWO PATHS BUILD IN DIFFERENT PLACES: the identical case creates its constant wherever the
/// PASS's builder points — [`ConstSink`] — and only the mapping path moves to the op (`:801-802`) and
/// puts the sink back afterwards (`:836`).
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
    const_sink: &mut ConstSink,
    values: &mut Values,
) -> Val {
    let first_flat_expr = flat_exprs.first().map(FlatExpr::first).unwrap_or(&[]);
    let first_flat_expr_size = first_flat_expr.len();
    if flat_exprs.len() == 1
        || are_flat_exprs_and_args_identical(flat_exprs, indices, result_info, operand_idx)
    {
        match first_flat_expr_size {
            1 => return sink_constant(scope, const_sink, first_flat_expr[0], ty, values),
            2 if operand_idx == 1 => {
                return sink_constant(scope, const_sink, first_flat_expr[operand_idx], ty, values);
            }
            // `getFirstExprInfo()->propagated_args_[operand_idx]` (`:790`, `:797`) — the first
            // NON-NULL list entry, which this branch has just proved every unit agrees with.
            2 | 3 => {
                if let Some(arg) = first_propagated_arg(expr_prop_analysis, result_info, operand_idx)
                {
                    return arg;
                }
            }
            _ => {}
        }
    }
    if units.len() != flat_exprs.len() {
        todo!(
            "createNewOpOrMap: DT_CHECK_MSG(units.size() == num_flat_exprs, \"Expecting a flattened \
             expression for each unit\") (ScalarSimplifications.cpp:805) — {} units against {} \
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

/// `result_info->getFirstExprInfo()->propagated_args_[operand_idx]` (`:790`, `:797`) — the identical
/// case's read, and `None` where the reference indexes a `SmallVector` past its end.
fn first_propagated_arg(
    expr_prop_analysis: &mut impl PropagationAnalysis,
    result_info: ExprInfoMap,
    operand_idx: usize,
) -> Option<Val> {
    expr_prop_analysis
        .first_propagated_args(result_info)
        .get(operand_idx)
        .copied()
}

/// `result_info->getExprInfoAt(at)->propagated_args_[operand_idx]` (`:816-817`, `:824-825`) — the
/// per-unit read, and `None` where the reference indexes a `SmallVector` past its end.
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

/// `sentient::ConstantOp::create(builder, loc, operand_type, value)` ON THE PASS'S OWN BUILDER
/// (`:786-788`, `:793-795`) — at the sink, which then advances past what it just created.
fn sink_constant(
    scope: &mut Vec<Op>,
    const_sink: &mut ConstSink,
    value: i64,
    ty: ScalarTy,
    values: &mut Values,
) -> Val {
    let at = const_sink.0.min(scope.len());
    let result = scalar_constant(scope, at, value, ty, values);
    const_sink.0 = at + 1;
    result
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

/// `sentient::ConstantOp` OR A CONSTANT `uniform::QueryMapOp`'S PER-UNIT TARGETS (`:618-626`,
/// `:634-643`) — the two spellings of an add's constant operand, EMPTY where it is neither, which is
/// the reference's `rhs_consts.empty()` refusal (`:627`, `:644`).
fn add_operand_constants(val: Val, defs: Definitions<'_>) -> Vec<i64> {
    match defs.of(val) {
        Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => vec![*value],
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            if !is_constant(val, ConstKind::ScalarConstant, defs) {
                todo!(
                    "lightWeightSimplifyBinaryArithmetic: \
                     DT_CHECK_MSG(isConstant<sentient::ConstantOp>(query_map), \"Expecting \
                     non-symbolic RHS of IfOp condition\") (ScalarSimplifications.cpp:623, :639)"
                )
            }
            constant_target_values(*map, defs)
        }
        _ => Vec::new(),
    }
}

/// Replaces: e534_lightWeightSimplifyBinaryArithmetic
///
/// Deletes an add of zero, folds a scalar add whose expression resolved to a constant into that
/// constant, and otherwise merges a chain of two adds into one over the sum of their constants.
///
/// ⛔ ONLY AN ADD LOSES ITS ZERO OPERAND (`:512-536`): `checkOperand` is written for both ops and
/// called for neither `sentient.scalar_sub`, so `x - 0` survives.
/// ⛔ THE CONSTANT ARM DELETES THE OP EVEN WITH NO USES (`:578-588`) — the replacement value is what
/// is conditional, not the deletion.
/// ⛔ e534'S FLATTENING DISCARDS ITS OWN FAILURE (`void(...)`, `:568`) where e532 and e533 decline on
/// it, so a failed unit still pushes an expression.
/// ⛔ THE MERGE ARM READS THE INNER ADD'S SECOND OPERAND ONLY (`:629`), keeps the outer op and
/// deletes the INNER one, and needs it to have exactly one use.
/// ⭐ THE PER-UNIT CONSTANT LISTS BROADCAST: a list of length 1 pairs with every element of the other
/// (`:650-659`), which is why the sum is indexed by `min(idx, len - 1)`.
pub(crate) fn light_weight_simplify_binary_arithmetic(
    scope: &mut Vec<Op>,
    at: usize,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    unit_index_map: &impl UnitIndexMap,
    to_be_deleted: &mut Vec<Val>,
    const_sink: &mut ConstSink,
    values: &mut Values,
) {
    let (is_add, result, ty) = match scope.get(at) {
        Some(Op::Sentient(ops::Op::ScalarAdd { result, ty, .. })) => (true, *result, *ty),
        Some(Op::Sentient(ops::Op::ScalarSub { result, ty, .. })) => (false, *result, *ty),
        _ => return,
    };
    let inputs = operands(&scope[at]);
    if is_add {
        for i in 0..inputs.len() {
            let zero = {
                let regions: [&[Op]; 1] = [scope];
                let defs = Definitions::from_innermost(&regions);
                matches!(
                    defs.of(inputs[i]),
                    Some(Op::Sentient(ops::Op::ScalarConstant { value: 0, .. }))
                )
            };
            if zero {
                replace_all_uses_with(scope, result, inputs[1 - i]);
                to_be_deleted.push(result);
                return;
            }
        }
    }
    let expr_info_map = expr_prop_analysis.affine_expression(result);
    if expr_prop_analysis.is_expr_info_map_empty(expr_info_map) {
        // A simplification that already ran can stale the analysis, or leave it without an entry for
        // an operand that simplification created.
        return;
    }
    // `op->emitOpError("can't find parentOp")` — ⭐ AND NO `signalPassFailure()`, unlike e532 and e533.
    let Some(QueryKeyAndUnits { key, mut units }) =
        super::query_key_and_units_from_parent_region(scope, at)
    else {
        return;
    };
    let mut indices = Vec::new();
    {
        let regions: [&[Op]; 1] = [scope];
        let defs = Definitions::from_innermost(&regions);
        select_indices_for_units(&mut units, &mut indices, unit_index_map, defs);
    }
    let unit_indices: Vec<usize> = indices.iter().map(|index| index.0 as usize).collect();
    let mut flat_exprs: Vec<FlatExpr> = Vec::new();
    let mut is_the_result_constant = true;
    for idx in &unit_indices {
        let Some(expr_info) = expr_prop_analysis.expr_info_at(expr_info_map, *idx) else {
            todo!(
                "lightWeightSimplifyBinaryArithmetic: DT_CHECK_MSG(expr_info, \"Expecting valid \
                 ExprInfo for unit\") (ScalarSimplifications.cpp:562) — no entry for unit {idx}"
            )
        };
        if !expr_info.cannot_be_resolved
            && expr_info.num_propagated_args == 0
            && expr_info.num_map_results == 1
        {
            let rows = expr_prop_analysis
                .flattened_affine_exprs(expr_info_map, *idx)
                .map(|flattened| flattened.rows)
                .unwrap_or_default();
            flat_exprs.push(FlatExpr(rows));
        } else {
            is_the_result_constant = false;
            break;
        }
    }
    if is_the_result_constant {
        if use_count(result, scope) > 0 {
            let new_op = create_new_op_or_map(
                scope,
                at,
                &flat_exprs,
                &unit_indices,
                &units,
                expr_info_map,
                0,
                ty,
                key,
                expr_prop_analysis,
                const_sink,
                values,
            );
            expr_prop_analysis.set_expr_info_map_for_value(new_op, expr_info_map);
            replace_all_uses_with(scope, result, new_op);
        }
        to_be_deleted.push(result);
        return;
    }
    // `x = ADD a, c0; y = ADD x, c1` --> `y = ADD a, c0 + c1`, which reads no propagated expression
    // at all: using one could disrupt what live range reduction did (`:594-600`).
    let Some(Op::Sentient(ops::Op::ScalarAdd { lhs, rhs, .. })) = scope.get(at) else {
        return;
    };
    let (lhs, rhs) = (*lhs, *rhs);
    let merge = {
        let regions: [&[Op]; 1] = [scope];
        let defs = Definitions::from_innermost(&regions);
        // `isa<BlockArgument>(lhs) || isa<BlockArgument>(rhs)`, and the inner op must be an add.
        let Some(Op::Sentient(ops::Op::ScalarAdd {
            lhs: inner_lhs,
            rhs: inner_rhs,
            result: inner_result,
            ..
        })) = defs.of(lhs)
        else {
            return;
        };
        if defs.of(rhs).is_none() {
            return;
        }
        // `if (rhs_consts.empty()) return;` (`:627`) IS BEFORE THE LHS IS EVEN LOOKED AT, so a
        // non-constant RHS never reaches the LHS `DT_CHECK_MSG` (`:639`).
        let rhs_consts = add_operand_constants(rhs, defs);
        if rhs_consts.is_empty() {
            return;
        }
        let inner_consts = add_operand_constants(*inner_rhs, defs);
        if inner_consts.is_empty() {
            return;
        }
        (*inner_lhs, *inner_result, inner_consts, rhs_consts)
    };
    let (inner_lhs, inner_result, inner_consts, rhs_consts) = merge;
    if use_count(inner_result, scope) != 1 {
        return;
    }
    let element_num = inner_consts.len().max(rhs_consts.len());
    if (inner_consts.len() != 1 && inner_consts.len() != element_num)
        || (rhs_consts.len() != 1 && rhs_consts.len() != element_num)
    {
        todo!(
            "lightWeightSimplifyBinaryArithmetic: DT_CHECK(lhs_add_op_consts.size() == 1 || \
             lhs_add_op_consts.size() == element_num) (ScalarSimplifications.cpp:652-654) — {} \
             against {} per-unit constants",
            inner_consts.len(),
            rhs_consts.len()
        )
    }
    let final_consts: Vec<i64> = (0..element_num)
        .map(|idx| {
            inner_consts[idx.min(inner_consts.len() - 1)] + rhs_consts[idx.min(rhs_consts.len() - 1)]
        })
        .collect();
    let before = scope.len();
    let new_operand = if element_num == 1 {
        scalar_constant(scope, at, final_consts[0], ty, values)
    } else {
        let mapped: Vec<Val> = final_consts
            .iter()
            .enumerate()
            .map(|(offset, value)| scalar_constant(scope, at + offset, *value, ty, values))
            .collect();
        let map = values.mint();
        let query = values.mint();
        scope.insert(
            at + element_num,
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: map,
                pairs: units.iter().copied().zip(mapped).collect(),
            }),
        );
        scope.insert(
            at + element_num + 1,
            Op::Uniform(uniform::Op::QueryMap {
                result: query,
                map,
                key,
            }),
        );
        query
    };
    let at = at + (scope.len() - before);
    set_operand(&mut scope[at], 0, inner_lhs);
    set_operand(&mut scope[at], 1, new_operand);
    to_be_deleted.push(inner_result);
}


/// WHETHER THE PROPAGATION ANALYSIS ANSWERED — `LogicalResult`, which here reports THE ANALYSIS'S
/// state and not a failure of this pass, so the crate's refusal ban is not in play.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) enum Propagation {
    /// `LogicalResult::success()` — the analysis ran, so the walk ran too.
    Succeeded,
    /// `LogicalResult::failure()` — `isPropagationSuccessful()` failed and nothing was simplified.
    Failed,
}

/// Replaces: e580_runOldLightWeightSimplifications
///
/// Hands every op of the block, and of the blocks nested in it, to
/// [`light_weight_simplify_binary_arithmetic`], then erases what the walk queued.
///
/// ⛔ A FAILED PROPAGATION ANALYSIS SIMPLIFIES NOTHING (`:691-693`), and the entry WIDENS TO THE
/// PARENT for `uniform.uniformize_regions`/`uniform.equalize_pattern` (`:696-698`) — the block this
/// takes is the widened one, which for every caller in the tree is a unit body (`:700`).
pub(crate) fn run_old_light_weight_simplifications(
    scope: &mut Vec<Op>,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    unit_index_map: &impl UnitIndexMap,
    values: &mut Values,
) -> Propagation {
    if !expr_prop_analysis.is_propagation_successful() {
        return Propagation::Failed;
    }
    let mut to_be_deleted = Vec::new();
    simplify_pre_order(
        scope,
        expr_prop_analysis,
        unit_index_map,
        &mut to_be_deleted,
        values,
    );
    for doomed in &to_be_deleted {
        erase_defining_op(scope, *doomed);
    }
    Propagation::Succeeded
}

/// `op->walk<WalkOrder::PreOrder>` — an op, then the blocks nested in it.
///
/// ⚠️ ONE [`ConstSink`] PER BLOCK where the reference has one builder for the whole nest, and e534
/// resolves an operand's definition inside the block it is given — so a nested op's constant lands at
/// the head of ITS block, and an operand defined further out is one e534 does not see.
fn simplify_pre_order(
    scope: &mut Vec<Op>,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    unit_index_map: &impl UnitIndexMap,
    to_be_deleted: &mut Vec<Val>,
    values: &mut Values,
) {
    let mut const_sink = ConstSink::default();
    let mut at = 0;
    while at < scope.len() {
        let before = scope.len();
        light_weight_simplify_binary_arithmetic(
            scope,
            at,
            expr_prop_analysis,
            unit_index_map,
            to_be_deleted,
            &mut const_sink,
            values,
        );
        // Everything e534 builds lands ahead of the op it simplified, so the op's index moves with it.
        at += scope.len() - before;
        for region in regions_mut(&mut scope[at]) {
            simplify_pre_order(region, expr_prop_analysis, unit_index_map, to_be_deleted, values);
        }
        at += 1;
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        ConstSink, FlatExpr, Propagation, create_new_op_or_map,
        light_weight_simplify_binary_arithmetic, run_old_light_weight_simplifications,
    };
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Op, Val, operands, sentient as ops};
    use crate::transform::sentient::analyses::{
        ExprInfoMap, OutOfScopePropagationAnalysis, OutOfScopeUnitIndexMap, PropagationAnalysis,
    };

    /// AN ANALYSIS THAT ANSWERS WHAT THE TEST SAYS — `PropagationAnalysis` is out of campaign scope.
    struct StatedAnalysis;

    impl PropagationAnalysis for StatedAnalysis {
        fn are_expressions_same(&mut self, _val1: Val, _val2: Val) -> bool {
            todo!("e472 never asks whether two expressions are the same")
        }

        fn propagated_args(&mut self, _map: ExprInfoMap, at: usize) -> Vec<Val> {
            vec![Val(50 + at as u32), Val(60 + at as u32)]
        }

        fn first_propagated_args(&mut self, _map: ExprInfoMap) -> Vec<Val> {
            vec![Val(50), Val(60)]
        }
    }

    /// AN ANALYSIS THAT RAN — e580's gate answers, and its zero-operand arm consults nothing else.
    struct PropagatingAnalysis;

    impl PropagationAnalysis for PropagatingAnalysis {
        fn are_expressions_same(&mut self, _val1: Val, _val2: Val) -> bool {
            todo!("e580 never asks whether two expressions are the same")
        }

        fn is_propagation_successful(&mut self) -> bool {
            true
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
    /// `sentient.scalar_constant` at the pass's sink, and one of length 3 answers with the propagated
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
                &mut ConstSink::default(),
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
                &mut ConstSink(1),
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
            &mut ConstSink::default(),
            &mut values_after(10),
        );
    }

    /// e472 — ⛔ THE IDENTICAL CASE BUILDS AT THE PASS'S OWN BUILDER (`:786-788`) AND NOT IN FRONT OF
    /// THE OP: the sink starts at the head of the block, so the constant lands at index 0 while the op
    /// it feeds sits at index 2, and the sink then advances past what it created.
    #[test]
    fn an_identical_case_constant_lands_at_the_pass_sink_not_before_the_op() {
        let mut scope = vec![
            Op::Sentient(ops::Op::Nop { dbg_name: None }),
            Op::Sentient(ops::Op::Nop { dbg_name: None }),
            Op::Sentient(ops::Op::Nop { dbg_name: None }),
        ];
        let mut const_sink = ConstSink::default();
        assert_eq!(
            create_new_op_or_map(
                &mut scope,
                2,
                &[FlatExpr(vec![vec![7]])],
                &[0],
                &[Val(80)],
                ExprInfoMap(0),
                0,
                ScalarTy::Index,
                Val(81),
                &mut StatedAnalysis,
                &mut const_sink,
                &mut values_after(10),
            ),
            Val(10)
        );
        assert!(matches!(
            scope[0],
            Op::Sentient(ops::Op::ScalarConstant { value: 7, .. })
        ));
        assert_eq!(const_sink, ConstSink(1));
    }

    /// e534 — the vendor's own case: an add of a zero constant is replaced by its other operand and
    /// queued for deletion, with the out-of-scope analyses proving that arm consults neither.
    #[test]
    fn an_add_of_a_zero_constant_is_replaced_by_its_other_operand() {
        let mut scope = vec![
            Op::Sentient(ops::Op::ScalarConstant {
                value: 0,
                result: Val(2),
                reg_locale: ops::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(ops::Op::ScalarAdd {
                lhs: Val(1),
                rhs: Val(2),
                result: Val(3),
                reg: None,
                element_size: None,
                ty: ScalarTy::Index,
            }),
            Op::Sentient(ops::Op::ScalarMul {
                lhs: Val(3),
                rhs: Val(1),
                result: Val(4),
                reg_locale: None,
                ty: ScalarTy::Index,
            }),
        ];
        let mut to_be_deleted = Vec::new();
        light_weight_simplify_binary_arithmetic(
            &mut scope,
            1,
            &mut OutOfScopePropagationAnalysis,
            &OutOfScopeUnitIndexMap,
            &mut to_be_deleted,
            &mut ConstSink::default(),
            &mut Values::default(),
        );
        assert_eq!(operands(&scope[2]), vec![Val(1), Val(1)]);
        assert_eq!(to_be_deleted, vec![Val(3)]);
    }

    /// e580 — the pre-order walk reaches an add of zero inside a `sentient.for` body, and the erase
    /// runs after the walk rather than under it.
    #[test]
    fn e580_simplifies_a_nested_add_of_zero_and_erases_it_afterwards() {
        let mut scope = vec![
            Op::Sentient(ops::Op::Nop { dbg_name: None }),
            Op::Sentient(ops::Op::For {
                iv: Val(3),
                bound: Val(0),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![
                    Op::Sentient(ops::Op::ScalarConstant {
                        value: 0,
                        result: Val(2),
                        reg_locale: ops::RegType::Imm,
                        ty: ScalarTy::Index,
                        is_symbol: false,
                    }),
                    Op::Sentient(ops::Op::ScalarAdd {
                        lhs: Val(1),
                        rhs: Val(2),
                        result: Val(4),
                        reg: None,
                        element_size: None,
                        ty: ScalarTy::Index,
                    }),
                    Op::Sentient(ops::Op::ScalarMul {
                        lhs: Val(4),
                        rhs: Val(1),
                        result: Val(5),
                        reg_locale: None,
                        ty: ScalarTy::Index,
                    }),
                ],
            }),
        ];
        assert_eq!(
            run_old_light_weight_simplifications(
                &mut scope,
                &mut PropagatingAnalysis,
                &OutOfScopeUnitIndexMap,
                &mut Values::default(),
            ),
            Propagation::Succeeded
        );
        let Op::Sentient(ops::Op::For { body, .. }) = &scope[1] else {
            panic!("the loop is still there")
        };
        assert_eq!(body.len(), 2);
        assert_eq!(operands(&body[1]), vec![Val(1), Val(1)]);
    }

    /// e580 — ⛔ A FAILED PROPAGATION ANALYSIS SIMPLIFIES NOTHING (`:691-693`): the same add of zero
    /// survives, which is why the walk cannot be reached through the out-of-scope analysis at all.
    #[test]
    fn e580_walks_nothing_when_the_propagation_analysis_failed() {
        struct NoPropagation;

        impl PropagationAnalysis for NoPropagation {
            fn are_expressions_same(&mut self, _val1: Val, _val2: Val) -> bool {
                todo!("e580 never asks whether two expressions are the same")
            }

            fn is_propagation_successful(&mut self) -> bool {
                false
            }
        }

        let mut scope = vec![Op::Sentient(ops::Op::ScalarAdd {
            lhs: Val(1),
            rhs: Val(2),
            result: Val(3),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })];
        let before = scope.clone();
        assert_eq!(
            run_old_light_weight_simplifications(
                &mut scope,
                &mut NoPropagation,
                &OutOfScopeUnitIndexMap,
                &mut Values::default(),
            ),
            Propagation::Failed
        );
        assert_eq!(scope, before);
    }
}
