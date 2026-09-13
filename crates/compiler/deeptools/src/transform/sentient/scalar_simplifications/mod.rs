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

//! `ScalarSimplifications.cpp` — 6 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e179_updateCmpIPredicate` | 179 | 0 | 8 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:63` |
//! | `e371_transformIfCondition` | 371 | 1 | 109 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:77` |
//! | `e471_simplifyConditionals` | 471 | 2 | 118 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:194` |
//! | `e532_simplifyBinaryOperation` | 532 | 3 | 89 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:315` |
//! | `e533_simplifyLoadStoreOperation` | 533 | 3 | 93 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:408` |
//! | `e581_runOnOperation` | 581 | 4 | 54 | `dcc/src/Transform/Sentient/ScalarSimplifications.cpp:872` |

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e581_runOnOperation` is the entry, and nothing
// runs it until a pass driver schedules it. CI runs clippy with `-D warnings`, so without this the
// items below fail the gate. ⭐ REMOVE THIS WITH THAT DRIVER, not with an anchor.

pub(crate) mod sentient;

use super::analyses::{InstructionEstimator, OutOfScopeInstructionEstimator};
use super::register_type_assignment::is_symbol;
use super::utils::select_indices_for_units;
use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::sentient::CmpPredicate;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, erase_defining_op, operands, regions_mut, replace_all_uses_with, results,
    sentient as ops, set_operand,
};
use crate::model::Model;
use crate::transform::sentient::analyses::{PropagationAnalysis, UnitIndexMap};
use crate::workload::Workload;

/// THE `$predicate` SLOT OF ONE `sentient.if` — `IfOp& if_op` reduced to the single field
/// `updateCmpIPredicate` writes, so "not a `sentient.if`" is not a case it has to answer.
///
/// ⛔ THE DIALECT IS IMPORTED AS `ops` HERE because this module already owns a `sentient` submodule.
pub struct IfPredicate<'a>(&'a mut CmpPredicate);

impl<'a> IfPredicate<'a> {
    /// The predicate slot of `op`, or `None` when `op` is not a `sentient.if`.
    #[must_use]
    pub fn of(op: &'a mut Op) -> Option<IfPredicate<'a>> {
        match op {
            Op::Sentient(ops::Op::If { predicate, .. }) => Some(IfPredicate(predicate)),
            _ => None,
        }
    }

    /// `if_op.getPredicate()`.
    #[must_use]
    pub fn get(&self) -> CmpPredicate {
        *self.0
    }
}

/// Replaces: e179_updateCmpIPredicate
///
/// Rewrites a `sentient.if`'s comparison to `new_predicate`, but only where it is an ORDERING.
///
/// ⛔ TRAP: `eq` AND `ne` ARE LEFT ALONE (`:65-66` names only `sgt`, `sge`, `slt`, `sle`) — the caller
/// e371 rewrites the operands either way, so an untouched `eq` is a reversed-operand `eq`, which is
/// the same test. Applying the new predicate to `ne` too would invert one.
/// ⭐ `OpBuilder& builder` IS UNREAD, and `CmpIPredicateAttr::get(getContext(), …)` is attribute
/// uniquing — both droppable mechanism for reaching the one field.
pub fn update_cmp_i_predicate(if_op: IfPredicate<'_>, new_predicate: CmpPredicate) {
    if matches!(
        if_op.get(),
        CmpPredicate::Sgt | CmpPredicate::Sge | CmpPredicate::Slt | CmpPredicate::Sle
    ) {
        *if_op.0 = new_predicate;
    }
}

// crustify:todo: e371_transformIfCondition
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:77  (109 body lines, level 1)
//   original  : void ScalarSimplificationsPass::transformIfCondition( IfOp& if_op, OpBuilder& builder, const affine::FlatAffineValueConstraints& constraints)
//   calls     : e179_updateCmpIPredicate, e252_size

/// Replaces: e471_simplifyConditionals
///
/// Turns one `sentient.if` into the single affine constraint `lhs - rhs {>=,>,==} 0` over both sides'
/// propagated arguments, and hands it to the solver that decides whether the branch can go.
///
/// ⛔ THE ARM ORDER IS THE PORT: `sle`/`slt` SWAP the two sides — of the subtraction AND of `all_args`
/// — and `sgt`/`slt` subtract a further 1, which is the strict comparison. `eq`/`ne` become an
/// EQUALITY where the four orderings become an inequality (`:268-270`).
/// ⛔ `else { return; }` (`:249`) IS UNREACHABLE HERE AND THAT IS A TYPE FACT: it covers MLIR's
/// unsigned `CmpIPredicate`s, and [`CmpPredicate`] is the six signed ones the dialect carries.
/// ⛔ BOTH `getAffineExpression` CALLS HAPPEN BEFORE EITHER `empty()` TEST (`:198-205`) — the analysis
/// memoises, so asking is not free of effect.
pub fn simplify_conditionals(
    expr_prop_analysis: &mut impl PropagationAnalysis,
    if_op: &mut Op,
    const_scope: &mut Vec<Op>,
) {
    let Op::Sentient(ops::Op::If {
        predicate,
        lhs,
        rhs,
        ..
    }) = &*if_op
    else {
        return;
    };
    let (predicate, lhs, rhs) = (*predicate, *lhs, *rhs);
    let lhs_info = expr_prop_analysis.affine_expression(lhs);
    let rhs_info = expr_prop_analysis.affine_expression(rhs);
    if expr_prop_analysis.is_expr_info_map_empty(lhs_info)
        || expr_prop_analysis.is_expr_info_map_empty(rhs_info)
    {
        // A simplification that already ran can stale the analysis, or leave it without an entry for
        // an operand that simplification created.
        return;
    }
    // `getFirstExprInfo()` on each side (`:221-222`), which is the FIRST NON-NULL list entry and not
    // unit index 0 — uniformization supports one DSC, so every unit's conditional is the same
    // (`:207-209`).
    let (minuend, subtrahend, strict) = match predicate {
        CmpPredicate::Sge | CmpPredicate::Eq | CmpPredicate::Ne => (lhs_info, rhs_info, false),
        CmpPredicate::Sle => (rhs_info, lhs_info, false),
        CmpPredicate::Sgt => (lhs_info, rhs_info, true),
        CmpPredicate::Slt => (rhs_info, lhs_info, true),
    };
    let mut all_args: Vec<Val> = expr_prop_analysis.first_propagated_args(minuend);
    all_args.extend(expr_prop_analysis.first_propagated_args(subtrahend));
    let equality = matches!(predicate, CmpPredicate::Eq | CmpPredicate::Ne);
    let _ = (strict, equality, all_args, const_scope);
    todo!(
        "affine::getFlattenedAffineExpr and affine::FlatAffineValueConstraints \
         (ScalarSimplifications.cpp:253-306) — out of campaign scope, and with them the local-variable \
         refusal, the redundant-constraint simplification and the one-constraint case that reaches \
         e371_transformIfCondition"
    )
}

/// WHAT `getQueryKeyAndUnitsFromParentRegion` FILLS IN — the uniformization key one op's enclosing
/// region is keyed by, and that region's unit list.
///
/// ⛔ THE REFERENCE LEAVES `key` NULL where a `dataflow.program_unit`'s region binds no argument
/// (`Utils.cpp:46-47`) and hands that null on to a `uniform.query_map`; representing that case is
/// e392's to settle, and this type's `None` is only its `LogicalResult::failure()`.
pub(super) struct QueryKeyAndUnits {
    /// `key` — the region argument every `uniform.query_map` this file builds is read through.
    pub(super) key: Val,
    /// `units` — the enclosing region's `$units`, before groups are expanded.
    pub(super) units: Vec<Val>,
}

/// `getQueryKeyAndUnitsFromParentRegion(op, key, units)` — e392, level 1, not yet ported.
///
/// ⛔ IT IS e392 AND IT IS NOT PORTED YET, and porting it will WIDEN THIS SIGNATURE: it climbs
/// `getParentRegion()` to the `dataflow.program_unit`, `uniform.uniformize_regions` or
/// `uniform.equalize_pattern` that binds the key, which one block does not carry — the enclosing path
/// ([`crate::transform::sentient::utils::OpAt`]) is what it needs.
pub(super) fn query_key_and_units_from_parent_region(
    block: &[Op],
    at: usize,
) -> Option<QueryKeyAndUnits> {
    let _ = (block, at);
    todo!(
        "getQueryKeyAndUnitsFromParentRegion (senpass e392, Utils.cpp:40) is not ported yet — the \
         climb to the enclosing region's uniformization key and unit list"
    )
}

/// WHERE THE PASS'S OWN `OpBuilder&` POINTS — `OpBuilder const_builder(unit);
/// const_builder.setInsertionPointToStart(unit.getBody(0))` (`:889-890`, `:902`; `:700` in e580), as
/// an insertion index into the block being walked that each op created through it advances past.
///
/// ⛔ IT IS NOT THE SIMPLIFIED OP'S OWN POSITION: only the mapping path of
/// [`sentient::create_new_op_or_map`] moves the builder to the op (`:801-802`) and puts it back
/// (`:836`) — an identical case's constant lands HERE, ahead of every op the walk has still to reach,
/// which is a longer live range and not just a different print order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConstSink(pub usize);

/// Replaces: e532_simplifyBinaryOperation
///
/// Rewrites one scalar add or sub whose propagated expression flattened to something simpler: a
/// constant replaces it outright, and a one- or two-variable expression re-points its operands.
///
/// ⛔ MORE THAN TWO PROPAGATED ARGUMENTS, MORE THAN ONE MAP RESULT, A FLATTENING FAILURE OR ANY LOCAL
/// VARIABLE ABANDONS THE WHOLE OP (`:344-359`) — including the units already flattened.
/// ⛔ LENGTH 3 IS AN ADD AND THE REFERENCE `DT_CHECK`s IT (`:382`), so a sub whose expression flattens
/// to two variables is a crash there and a named `todo!` here.
/// ⛔ LENGTH 2 SPLITS ON THE CONSTANT TERM: non-zero re-points both operands, ZERO replaces the op
/// with its variable (`:385-402`).
/// ⛔ A LENGTH THAT IS NEITHER 1 NOR 3 READS `[1]`, so an empty expression is the reference's own
/// out-of-bounds read — [`sentient::FlatExpr::first`] answers the empty slice and 0 takes the
/// replace arm.
/// ⭐ EVERY VALUE THIS BUILDS LANDS AHEAD OF THE OP — at [`ConstSink`] or immediately before it — so
/// the op's own index walks forward as they arrive.
pub fn simplify_binary_operation(
    scope: &mut Vec<Op>,
    at: usize,
    operand_type: ScalarTy,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    unit_index_map: &impl UnitIndexMap,
    to_be_deleted: &mut Vec<Val>,
    const_sink: &mut ConstSink,
    values: &mut Values,
) {
    let Some(result) = scope.get(at).and_then(|op| results(op).first().copied()) else {
        return;
    };
    let expr_info_map = expr_prop_analysis.affine_expression(result);
    if expr_prop_analysis.is_expr_info_map_empty(expr_info_map) {
        // A simplification that already ran can stale the analysis, or leave it without an entry for
        // an operand that simplification created.
        return;
    }
    // `op->emitOpError("can't find parentOp"); signalPassFailure();`
    let Some(QueryKeyAndUnits { key, mut units }) =
        query_key_and_units_from_parent_region(scope, at)
    else {
        return;
    };
    let mut indices = Vec::new();
    {
        let regions: [&[Op]; 1] = [scope];
        select_indices_for_units(
            &mut units,
            &mut indices,
            unit_index_map,
            Definitions::from_innermost(&regions),
        );
    }
    let unit_indices: Vec<usize> = indices.iter().map(|index| index.0 as usize).collect();
    let mut flat_exprs: Vec<sentient::FlatExpr> = Vec::new();
    for idx in &unit_indices {
        let Some(expr_info) = expr_prop_analysis.expr_info_at(expr_info_map, *idx) else {
            todo!(
                "simplifyBinaryOperation: DT_CHECK_MSG(expr_info, \"Expecting valid ExprInfo for \
                 unit\") (ScalarSimplifications.cpp:344) — no entry for unit {idx}"
            )
        };
        if expr_info.num_propagated_args > 2 || expr_info.num_map_results > 1 {
            return;
        }
        let Some(flattened) = expr_prop_analysis.flattened_affine_exprs(expr_info_map, *idx) else {
            return;
        };
        // MLIR cannot construct a local variable without an explicit representation.
        if flattened.num_local_vars > 0 {
            return;
        }
        flat_exprs.push(sentient::FlatExpr(flattened.rows));
    }
    let valid = {
        let regions: [&[Op]; 1] = [scope];
        sentient::are_all_exprs_valid_to_transform(
            &flat_exprs,
            &unit_indices,
            expr_info_map,
            sentient::ExprUse::BinaryOperation,
            expr_prop_analysis,
            Definitions::from_innermost(&regions),
        )
    };
    if !valid {
        return;
    }
    let first_coeffs: Vec<i64> = flat_exprs
        .first()
        .map_or_else(Vec::new, |flat_expr| flat_expr.first().to_vec());
    let new_operand = |scope: &mut Vec<Op>,
                           at: usize,
                           operand_idx: usize,
                           expr_prop_analysis: &mut _,
                           const_sink: &mut ConstSink,
                           values: &mut Values| {
        let before = scope.len();
        let operand = sentient::create_new_op_or_map(
            scope,
            at,
            &flat_exprs,
            &unit_indices,
            &units,
            expr_info_map,
            operand_idx,
            operand_type,
            key,
            expr_prop_analysis,
            const_sink,
            values,
        );
        (operand, scope.len() - before)
    };
    match first_coeffs.len() {
        // A constant: the op goes.
        1 => {
            let (operand, _) = new_operand(scope, at, 0, expr_prop_analysis, const_sink, values);
            expr_prop_analysis.set_expr_info_map_for_value(operand, expr_info_map);
            replace_all_uses_with(scope, result, operand);
            to_be_deleted.push(result);
        }
        // `1 * variable + 1 * variable + constant` — the constant has to be zero, so both operands
        // are rewritten and the add stays.
        3 => {
            let (operand0, grew0) =
                new_operand(scope, at, 0, expr_prop_analysis, const_sink, values);
            let (operand1, grew1) =
                new_operand(scope, at + grew0, 1, expr_prop_analysis, const_sink, values);
            let at = at + grew0 + grew1;
            if !matches!(scope.get(at), Some(Op::Sentient(ops::Op::ScalarAdd { .. }))) {
                todo!(
                    "simplifyBinaryOperation: DT_CHECK(isa<sentient::AddOp>(op)) \
                     (ScalarSimplifications.cpp:382) — a two-variable expression on {:?}",
                    scope.get(at)
                )
            }
            set_operand(&mut scope[at], 0, operand0);
            set_operand(&mut scope[at], 1, operand1);
        }
        // `1 * variable + constant` — the variable's coefficient has to be 1.
        _ => {
            if first_coeffs.get(1).copied().unwrap_or(0) != 0 {
                let (operand0, grew0) =
                    new_operand(scope, at, 0, expr_prop_analysis, const_sink, values);
                let (operand1, grew1) =
                    new_operand(scope, at + grew0, 1, expr_prop_analysis, const_sink, values);
                let at = at + grew0 + grew1;
                set_operand(&mut scope[at], 0, operand0);
                set_operand(&mut scope[at], 1, operand1);
            } else {
                let (operand0, _) =
                    new_operand(scope, at, 0, expr_prop_analysis, const_sink, values);
                replace_all_uses_with(scope, result, operand0);
                to_be_deleted.push(result);
            }
        }
    }
}

/// THE OPERAND SLOTS `simplifyLoadStoreOperation` REWRITES (`:470-501`) — ⭐ THE ADDRESS PAIRS AND
/// NOTHING ELSE: an increment, a `dst`, a multicast operand or an element index is never simplified.
///
/// ⛔ `sentient.load_and_store` HAS ITS TWO PAIRS APART: `src`, `dst` and both increments sit between
/// and around them in the operand order [`operands`] answers.
fn address_slots(op: &Op) -> &'static [usize] {
    match op {
        Op::Sentient(
            ops::Op::LoadAndSend { .. }
            | ops::Op::ReceiveAndStore { .. }
            | ops::Op::LoadAndExtractScalar { .. }
            | ops::Op::LoadComputeAndSend { .. },
        ) => &[0, 1],
        Op::Sentient(ops::Op::LoadAndStore { .. }) => &[2, 3, 5, 6],
        _ => &[],
    }
}

/// Replaces: e533_simplifyLoadStoreOperation
///
/// Replaces every address operand of one transfer with the simplest value its propagated expression
/// allows — the variable it is written over, or a fresh constant.
///
/// ⛔ A SYMBOL IS NEVER SIMPLIFIED (`:425`) and neither is an operand whose expression is stale, has
/// more than one propagated argument, more than one dimension, fails to flatten or needs a local
/// variable — each of those leaves the operand exactly as it was.
/// ⛔ ONE DIMENSION ANSWERS WITH THE PROPAGATED ARGUMENT ITSELF (`:456-459`), which is the whole point
/// of the load/store flavour of e372: no op is created for it.
/// ⛔ THE UNIT LIST AND KEY ARE READ BEFORE THE OP KIND IS (`:412-420`), so a transfer this cannot
/// name still reports `can't find parentOp` from an unrooted op.
/// ⭐ EVERY ADDRESS OPERAND IS `Index` BY THE OP'S OWN DECLARATION (`SentientOps.td:460`, `:507`,
/// `:550`, `:609`), which is what `val.getType()` hands `createNewOpOrMap`.
pub fn simplify_load_store_operation(
    scope: &mut Vec<Op>,
    at: usize,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    unit_index_map: &impl UnitIndexMap,
    const_sink: &mut ConstSink,
    values: &mut Values,
) {
    // `op->emitOpError("can't find parentOp"); signalPassFailure();`
    let Some(QueryKeyAndUnits { key, mut units }) =
        query_key_and_units_from_parent_region(scope, at)
    else {
        return;
    };
    let mut indices = Vec::new();
    {
        let regions: [&[Op]; 1] = [scope];
        select_indices_for_units(
            &mut units,
            &mut indices,
            unit_index_map,
            Definitions::from_innermost(&regions),
        );
    }
    let unit_indices: Vec<usize> = indices.iter().map(|index| index.0 as usize).collect();
    let Some(slots) = scope.get(at).map(address_slots) else {
        return;
    };
    let mut at = at;
    for slot in slots {
        let Some(val) = operands(&scope[at]).get(*slot).copied() else {
            continue;
        };
        let before = scope.len();
        let simplified = simplified_value(
            scope,
            at,
            val,
            &unit_indices,
            &units,
            key,
            expr_prop_analysis,
            const_sink,
            values,
        );
        at += scope.len() - before;
        if let Some(simplified) = simplified {
            set_operand(&mut scope[at], *slot, simplified);
        }
    }
}

/// `getSimplifiedValue(Value)` (`:422-468`) — the value one address operand can be replaced by, or
/// `None` for the reference's empty `Value()`, which leaves the operand alone.
fn simplified_value(
    scope: &mut Vec<Op>,
    at: usize,
    val: Val,
    unit_indices: &[usize],
    units: &[Val],
    key: Val,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    const_sink: &mut ConstSink,
    values: &mut Values,
) -> Option<Val> {
    {
        let regions: [&[Op]; 1] = [scope];
        if is_symbol(val, Definitions::from_innermost(&regions)) {
            return None;
        }
    }
    let expr_info_map = expr_prop_analysis.affine_expression(val);
    if expr_prop_analysis.is_expr_info_map_empty(expr_info_map) {
        return None;
    }
    let mut flat_exprs: Vec<sentient::FlatExpr> = Vec::new();
    for idx in unit_indices {
        let Some(expr_info) = expr_prop_analysis.expr_info_at(expr_info_map, *idx) else {
            todo!(
                "simplifyLoadStoreOperation: DT_CHECK_MSG(expr_info, \"Expecting valid ExprInfo for \
                 unit\") (ScalarSimplifications.cpp:436) — no entry for unit {idx}"
            )
        };
        if expr_info.num_propagated_args > 1 || expr_info.num_map_dims > 1 {
            return None;
        }
        let flattened = expr_prop_analysis.flattened_affine_exprs(expr_info_map, *idx)?;
        if flattened.num_local_vars > 0 {
            return None;
        }
        flat_exprs.push(sentient::FlatExpr(flattened.rows));
    }
    let valid = {
        let regions: [&[Op]; 1] = [scope];
        sentient::are_all_exprs_valid_to_transform(
            &flat_exprs,
            unit_indices,
            expr_info_map,
            sentient::ExprUse::LoadStore,
            expr_prop_analysis,
            Definitions::from_innermost(&regions),
        )
    };
    if !valid {
        return None;
    }
    // `getFirstExprInfo()` (`:456-457`) — either a direct variable or a constant. ⛔ THE FIRST
    // NON-NULL LIST ENTRY, NOT UNIT INDEX 0, and this read is not inside an identical-only branch.
    let first = expr_prop_analysis.first_expr_info(expr_info_map)?;
    if first.num_map_dims == 1 {
        return expr_prop_analysis.first_propagated_args(expr_info_map).first().copied();
    }
    Some(sentient::create_new_op_or_map(
        scope,
        at,
        &flat_exprs,
        unit_indices,
        units,
        expr_info_map,
        0,
        ScalarTy::Index,
        key,
        expr_prop_analysis,
        const_sink,
        values,
    ))
}

/// `-dcc-scalar-simplification-disable`, `cl::init(false)` (`:58-61`) — a `dcc-opt` command-line flag,
/// not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `opts_.OptLevel == 0` (`:884`) — the pipeline's optimisation level, which is `2` by default
/// (`dcc/tools/Options/dcc-pass-option.h:63-65`), so the shipped pipeline never reaches the
/// `haveIbuffSpace` half of the `&&`.
const OPT_LEVEL_ZERO: bool = false;

/// THE FIRST WALK — `unit->walk<PreOrder>` for `sentient::IfOp` (`:891-894`).
///
/// ⚠️ THE CONSTANTS GO TO A BLOCK OF THEIR OWN and are spliced in at the sink afterwards, because a
/// `sentient.if` nested in the body cannot be handed to [`simplify_conditionals`] while the body it
/// sits in is the const block. Creation order and the head-of-body position are the sink's own.
fn simplify_conditionals_pre_order(
    scope: &mut Vec<Op>,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    consts: &mut Vec<Op>,
) {
    for op in scope.iter_mut() {
        if matches!(op, Op::Sentient(ops::Op::If { .. })) {
            simplify_conditionals(expr_prop_analysis, op, consts);
        }
        for region in regions_mut(op) {
            simplify_conditionals_pre_order(region, expr_prop_analysis, consts);
        }
    }
}

/// THE SECOND WALK — `unit->walk<PreOrder>` over the arithmetic and the transfers (`:897-914`).
///
/// ⚠️ ONE [`ConstSink`] PER BLOCK where the reference has one for the whole unit: a nested op's
/// constant lands at the head of ITS block, which still dominates the use.
fn simplify_arithmetic_pre_order(
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
        match &scope[at] {
            Op::Sentient(ops::Op::ScalarAdd { ty, .. } | ops::Op::ScalarSub { ty, .. }) => {
                // `add_op.getInp1().getType()` (`:901`, `:906`) — the op's own type, which
                // `SameOperandsAndResultType` makes the first operand's.
                let operand_type = *ty;
                simplify_binary_operation(
                    scope,
                    at,
                    operand_type,
                    expr_prop_analysis,
                    unit_index_map,
                    to_be_deleted,
                    &mut const_sink,
                    values,
                );
            }
            Op::Sentient(
                ops::Op::LoadAndSend { .. }
                | ops::Op::ReceiveAndStore { .. }
                | ops::Op::LoadAndStore { .. }
                | ops::Op::LoadAndExtractScalar { .. }
                | ops::Op::LoadComputeAndSend { .. },
            ) => simplify_load_store_operation(
                scope,
                at,
                expr_prop_analysis,
                unit_index_map,
                &mut const_sink,
                values,
            ),
            _ => {}
        }
        // Everything the simplifications build lands ahead of the op, so its index moves with it.
        at += scope.len() - before;
        for region in regions_mut(&mut scope[at]) {
            simplify_arithmetic_pre_order(
                region,
                expr_prop_analysis,
                unit_index_map,
                to_be_deleted,
                values,
            );
        }
        at += 1;
    }
}

/// Replaces: e581_runOnOperation
///
/// The pass entry: simplify every conditional of every unit, then every scalar add, sub and transfer
/// address in it, and erase what the second walk queued.
///
/// ⛔ A FAILED PROPAGATION ANALYSIS DOES NOT STOP THE PASS (`:876-879`): there is no `return` after
/// `signalPassFailure()`, so both walks still run.
/// ⛔ THE MISSING ANALYSIS IS NAMED, NOT SUBSTITUTED FOR — `haveIbuffSpace` stays a `todo!` behind
/// [`OPT_LEVEL_ZERO`], which the pipeline fixes rather than this pass.
/// ⚠️ `markAnalysesPreserved<PropagationAnalysis>()` (`:917`) IS PASS-MANAGER BOOKKEEPING: no IR.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    expr_prop_analysis: &mut impl PropagationAnalysis,
    unit_index_map: &impl UnitIndexMap,
    values: &mut Values,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    // `module_op->emitError("Unable to perform expression propagation"); signalPassFailure();`
    let _propagation_succeeded = expr_prop_analysis.is_propagation_successful();
    for unit in program.units.iter_mut() {
        let mut instruction_estimator = OutOfScopeInstructionEstimator;
        if OPT_LEVEL_ZERO && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        let mut consts = Vec::new();
        simplify_conditionals_pre_order(&mut unit.body, expr_prop_analysis, &mut consts);
        unit.body.splice(0..0, consts);
        let mut to_be_deleted = Vec::new();
        simplify_arithmetic_pre_order(
            &mut unit.body,
            expr_prop_analysis,
            unit_index_map,
            &mut to_be_deleted,
            values,
        );
        for doomed in &to_be_deleted {
            erase_defining_op(&mut unit.body, *doomed);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        ConstSink, IfPredicate, Model, Program, Workload, run_on_operation,
        simplify_binary_operation, simplify_conditionals, simplify_load_store_operation,
        update_cmp_i_predicate,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units, Values};
    use crate::islands::sentient::dialects::sentient::CmpPredicate;
    use crate::islands::sentient::dialects::{Op, Val, sentient as ops};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::transform::sentient::analyses::{
        ExprInfoMap, OutOfScopeUnitIndexMap, PropagationAnalysis,
    };
    use crate::units::DfirUnit;

    /// A model, so the program is typed; nothing here reads it.
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

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// One program whose only unit holds `body`.
    fn program_with(body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit::<Dd2> {
                    on: Units::one(DfirUnit::Lxlu, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// A `sentient.for` around `body`, so a walk has somewhere to descend to.
    fn loop_over(body: Vec<Op>) -> Op {
        Op::Sentient(ops::Op::For {
            iv: Val(8),
            bound: Val(9),
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

    /// AN ANALYSIS THAT ANSWERS WHAT THE TEST SAYS — `PropagationAnalysis` is out of campaign scope,
    /// so what is under test is the EFFECT this unit has given an answer.
    struct StatedAnalysis {
        /// Whether every value's `ExprInfoMap` is empty.
        empty: bool,
    }

    impl PropagationAnalysis for StatedAnalysis {
        fn are_expressions_same(&mut self, _val1: Val, _val2: Val) -> bool {
            todo!("e471 never asks whether two expressions are the same")
        }

        fn is_propagation_successful(&mut self) -> bool {
            true
        }

        fn affine_expression(&mut self, val: Val) -> ExprInfoMap {
            ExprInfoMap(val.0)
        }

        fn is_expr_info_map_empty(&mut self, _map: ExprInfoMap) -> bool {
            self.empty
        }

        fn propagated_args(&mut self, map: ExprInfoMap, _at: usize) -> Vec<Val> {
            vec![Val(map.0)]
        }

        fn first_propagated_args(&mut self, map: ExprInfoMap) -> Vec<Val> {
            vec![Val(map.0)]
        }
    }

    /// `sentient.if %lhs `pred` %rhs { }`.
    fn if_op(predicate: CmpPredicate) -> Op {
        Op::Sentient(ops::Op::If {
            predicate,
            lhs: Val(0),
            rhs: Val(1),
            yielded: Vec::new(),
            dbg_name: None,
            then_body: Vec::new(),
            else_body: Vec::new(),
        })
    }

    /// e179 — an ordering is rewritten and `eq`/`ne` are left exactly as they were.
    #[test]
    fn only_an_ordering_predicate_is_rewritten() {
        for original in [
            CmpPredicate::Sgt,
            CmpPredicate::Sge,
            CmpPredicate::Slt,
            CmpPredicate::Sle,
        ] {
            let mut op = if_op(original);
            update_cmp_i_predicate(
                IfPredicate::of(&mut op).expect("a sentient.if"),
                CmpPredicate::Sle,
            );
            assert_eq!(
                IfPredicate::of(&mut op).map(|p| p.get()),
                Some(CmpPredicate::Sle)
            );
        }
        for untouched in [CmpPredicate::Eq, CmpPredicate::Ne] {
            let mut op = if_op(untouched);
            update_cmp_i_predicate(
                IfPredicate::of(&mut op).expect("a sentient.if"),
                CmpPredicate::Sle,
            );
            assert_eq!(IfPredicate::of(&mut op).map(|p| p.get()), Some(untouched));
        }
    }

    /// e471 — a stale analysis with no expression for an operand leaves the `sentient.if` exactly as
    /// it was, which is the reference's own reason for the `empty()` test.
    #[test]
    fn a_conditional_whose_operands_have_no_propagated_expression_is_left_alone() {
        let mut op = if_op(CmpPredicate::Sge);
        let untouched = op.clone();
        let mut consts = Vec::new();
        simplify_conditionals(&mut StatedAnalysis { empty: true }, &mut op, &mut consts);
        assert_eq!(op, untouched);
        assert!(consts.is_empty());
    }

    /// e471 — with expressions on both sides the unit reaches the affine solver, which is out of
    /// campaign scope.
    #[test]
    #[should_panic(expected = "out of campaign scope")]
    fn a_conditional_with_propagated_expressions_reaches_the_affine_solver() {
        let mut op = if_op(CmpPredicate::Sge);
        let mut consts = Vec::new();
        simplify_conditionals(&mut StatedAnalysis { empty: false }, &mut op, &mut consts);
    }

    /// e532 — a stale analysis with no expression for the result leaves the add exactly as it was and
    /// deletes nothing, which is the reference's own reason for the `empty()` test.
    #[test]
    fn a_binary_operation_whose_result_has_no_propagated_expression_is_left_alone() {
        let mut scope = vec![Op::Sentient(ops::Op::ScalarAdd {
            lhs: Val(1),
            rhs: Val(2),
            result: Val(3),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })];
        let untouched = scope.clone();
        let mut to_be_deleted = Vec::new();
        simplify_binary_operation(
            &mut scope,
            0,
            ScalarTy::Index,
            &mut StatedAnalysis { empty: true },
            &OutOfScopeUnitIndexMap,
            &mut to_be_deleted,
            &mut ConstSink::default(),
            &mut Values::default(),
        );
        assert_eq!(scope, untouched);
        assert!(to_be_deleted.is_empty());
    }

    /// e533 — the unit list is read before the op kind is, so even an op with no address operands at
    /// all reaches the unported parent-region query rather than returning early.
    #[test]
    #[should_panic(expected = "senpass e392")]
    fn a_load_store_asks_for_its_units_before_it_looks_at_the_op() {
        let mut scope = vec![Op::Sentient(ops::Op::Nop { dbg_name: None })];
        simplify_load_store_operation(
            &mut scope,
            0,
            &mut StatedAnalysis { empty: false },
            &OutOfScopeUnitIndexMap,
            &mut ConstSink::default(),
            &mut Values::default(),
        );
    }

    /// e581 — the entry runs both walks over the whole nest of every unit, and a stale analysis is
    /// what leaves the conditional and the arithmetic in it alone.
    #[test]
    fn e581_walks_every_units_nest_and_a_stale_analysis_rewrites_nothing() {
        let body = vec![
            loop_over(vec![
                if_op(CmpPredicate::Sge),
                Op::Sentient(ops::Op::ScalarAdd {
                    lhs: Val(1),
                    rhs: Val(2),
                    result: Val(3),
                    reg: None,
                    element_size: None,
                    ty: ScalarTy::Index,
                }),
            ]),
            Op::Sentient(ops::Op::Nop { dbg_name: None }),
        ];
        let mut program = program_with(body.clone());
        run_on_operation(
            &mut program,
            &mut StatedAnalysis { empty: true },
            &OutOfScopeUnitIndexMap,
            &mut Values::default(),
        );
        assert_eq!(program.units.iter().next().expect("the head unit").body, body);
    }

    /// e581 — the first walk reaches a `sentient.if` nested in a loop, and with expressions on both
    /// sides that conditional reaches the affine solver, which is out of campaign scope.
    #[test]
    #[should_panic(expected = "out of campaign scope")]
    fn e581_reaches_a_nested_conditional_through_the_first_walk() {
        let mut program = program_with(vec![loop_over(vec![if_op(CmpPredicate::Sge)])]);
        run_on_operation(
            &mut program,
            &mut StatedAnalysis { empty: false },
            &OutOfScopeUnitIndexMap,
            &mut Values::default(),
        );
    }
}
