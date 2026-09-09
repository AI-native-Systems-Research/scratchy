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
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e581_runOnOperation` (level 4) is what reaches
// the item below, which nothing but this file's tests calls until it lands. CI runs clippy with
// `-D warnings`. ⭐ REMOVE THIS WITH e581.

pub(crate) mod sentient;

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient::CmpPredicate;
use crate::islands::sentient::dialects::{Op, Val, sentient as ops};

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

/// THE WHOLE COMPARISON OF ONE `sentient.if` — the predicate plus the two operands e371 rewrites,
/// which is [`IfPredicate`] widened by exactly `setOperand(0)` and `setOperand(1)`.
pub struct IfCondition<'a> {
    /// `$predicate`.
    predicate: &'a mut CmpPredicate,
    /// `if_op->getOperand(0)`.
    lhs: &'a mut Val,
    /// `if_op->getOperand(1)`.
    rhs: &'a mut Val,
}

impl<'a> IfCondition<'a> {
    /// The comparison of `op`, or `None` when `op` is not a `sentient.if`.
    #[must_use]
    pub fn of(op: &'a mut Op) -> Option<IfCondition<'a>> {
        match op {
            Op::Sentient(ops::Op::If {
                predicate,
                lhs,
                rhs,
                ..
            }) => Some(IfCondition {
                predicate,
                lhs,
                rhs,
            }),
            _ => None,
        }
    }

    /// `if_op.getPredicate()`.
    #[must_use]
    pub fn get(&self) -> CmpPredicate {
        *self.predicate
    }

    /// The predicate slot alone, which is what [`update_cmp_i_predicate`] takes.
    fn predicate(&mut self) -> IfPredicate<'_> {
        IfPredicate(self.predicate)
    }
}

/// ONE DIMENSION OF THE ROW — `constraints.getValue(dim_id)` with the type the `DT_CHECK` at `:130-134`
/// compares and the coefficient the sort at `:89-101` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintDim {
    /// `constraints.getValue(dim_id)`.
    pub value: Val,
    /// `V.getType()` — what `arg_type` is taken from.
    pub ty: ScalarTy,
    /// `coefficients[dim_id]`.
    pub coeff: i64,
}

/// THE ONE ROW OF A `FlatAffineValueConstraints` THIS PASS READS (`:79-82`).
///
/// ⭐ EQUALITY-OR-INEQUALITY IS THE CALLER'S CHOICE: `getNumEqualities() == 1 ? getEquality(0) :
/// getInequality(0)` is decided where the counts are known, which is e471, and only the row arrives.
/// ⛔ THE SYMBOL AND LOCAL COLUMNS ARE NOT CARRIED — the loop stops at `getNumDimVars()` and the only
/// other column ever read is `coefficients.back()`, which is [`ConstraintRow::constant`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConstraintRow {
    /// One entry per dimension variable, in the row's own order.
    pub dims: Vec<ConstraintDim>,
    /// `coefficients.back()` — the row's constant term.
    pub constant: i64,
}

/// WHICH SIDE OF THE COMPARISON THE ROW'S CONSTANT LANDED ON (`:113-124`).
///
/// ⛔ THE TWO ARE MUTUALLY EXCLUSIVE, SO THIS IS ONE `Option` AND NOT TWO VECTORS:
/// `transformed_lhs_constant_arg` is filled only for a constant `>= 0` and
/// `transformed_rhs_constant_arg` only for one `< 0`. That makes the reference's both-constants arm
/// (`:167-175`) unreachable — along with the `transformed_rhs_constant_arg[0] +
/// transformed_rhs_constant_arg[0]` in it, which cannot be written here at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformedConst {
    /// `transformed_lhs_constant_arg` — the constant belongs on the LEFT.
    Lhs(i64),
    /// `transformed_rhs_constant_arg`, already negated as `:120` negates it.
    Rhs(i64),
}

/// Replaces: e371_transformIfCondition
///
/// Rewrites one `sentient.if`'s comparison into the two operands its constraint row states, creating a
/// constant for the row's constant term where a side needs one, and leaves the op alone otherwise.
///
/// ⛔ A NON-UNIT COEFFICIENT, THREE OPERANDS, OR ANY OTHER SHAPE IS A NO-OP — five of the reference's
/// exits are `return`s that have already changed nothing (`:100`, `:123`, `:177`, `:180`).
/// ⛔ `-1 >= 0` BECOMES `> 0` AND THE CONSTANT BECOMES ZERO (`:109-112`), which then takes the
/// non-negative branch below; `update_cmp_i_predicate` is what refuses to apply `sge` to an `eq`.
/// ⭐ THE `DT_CHECK` ON THE OPERAND TYPES (`:130-134`) IS THE CALLER'S INVARIANT: `arg_type` is one
/// dimension's, and a row mixing types is a broken constraint system, not an input.
pub fn transform_if_condition(
    mut if_op: IfCondition<'_>,
    row: &ConstraintRow,
    consts: &mut Vec<Op>,
    values: &mut Values,
) {
    let mut lhs_args: Vec<ConstraintDim> = Vec::new();
    let mut rhs_args: Vec<ConstraintDim> = Vec::new();
    for dim in &row.dims {
        match dim.coeff {
            1 => lhs_args.push(*dim),
            -1 => rhs_args.push(*dim),
            0 => {}
            // "We don't support multiplication with non-unit value in the simplification."
            _ => return,
        }
    }

    let lhs_rhs_arg_present = !lhs_args.is_empty() && !rhs_args.is_empty();
    let mut const_coeff = row.constant;
    let mut new_predicate = CmpPredicate::Sge;
    if const_coeff == -1 && if_op.get() == CmpPredicate::Sge {
        const_coeff = 0;
        new_predicate = CmpPredicate::Sgt;
    }

    let constant = if const_coeff >= 0 && !lhs_rhs_arg_present {
        Some(TransformedConst::Lhs(const_coeff))
    } else if const_coeff < 0 {
        Some(TransformedConst::Rhs(-const_coeff))
    } else if const_coeff == 0 {
        // Reaching here IS `lhs_rhs_arg_present` (`:122`); a non-zero constant beside both sides would
        // need a third operand.
        None
    } else {
        return;
    };

    let arg_type = lhs_args
        .last()
        .or_else(|| rhs_args.last())
        .map_or(ScalarTy::Int(32), |dim| dim.ty);

    // `sentient.if` supports only two variables at max!
    if lhs_args.len() + rhs_args.len() + usize::from(constant.is_some()) > 2 {
        return;
    }
    match (lhs_args.as_slice(), rhs_args.as_slice(), constant) {
        ([lhs], [rhs], None) => {
            *if_op.lhs = lhs.value;
            *if_op.rhs = rhs.value;
        }
        ([lhs], [], Some(TransformedConst::Lhs(const_arg))) => {
            *if_op.lhs = lhs.value;
            *if_op.rhs = constant_val(-const_arg, arg_type, consts, values);
        }
        ([lhs], [], Some(TransformedConst::Rhs(const_arg))) => {
            *if_op.lhs = lhs.value;
            *if_op.rhs = constant_val(const_arg, arg_type, consts, values);
        }
        // ⭐ THE REFERENCE'S TWO rhs-ARG ARMS ARE ONE ARM: both put the constant on the left, and
        // neither negates it (`:155-166`).
        ([], [rhs], Some(TransformedConst::Lhs(const_arg) | TransformedConst::Rhs(const_arg))) => {
            *if_op.lhs = constant_val(const_arg, arg_type, consts, values);
            *if_op.rhs = rhs.value;
        }
        _ => return,
    }
    update_cmp_i_predicate(if_op.predicate(), new_predicate);
}

/// `sentient::ConstantOp::create(builder, if_op->getLoc(), arg_type, value)` — one constant in the
/// const builder's own block, which is why it goes to `consts` and not beside the `sentient.if`.
fn constant_val(value: i64, ty: ScalarTy, consts: &mut Vec<Op>, values: &mut Values) -> Val {
    let result = values.mint();
    consts.push(Op::Sentient(ops::Op::ScalarConstant {
        value,
        result,
        reg_locale: ops::RegType::Imm,
        ty,
        is_symbol: false,
    }));
    result
}

// crustify:todo: e471_simplifyConditionals
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:194  (118 body lines, level 2)
//   original  : void ScalarSimplificationsPass::simplifyConditionals( PropagationAnalysis& expr_prop_analysis, sentient::IfOp& if_op, OpBuilder& const_builder)
//   calls     : e252_size, e371_transformIfCondition

// crustify:todo: e532_simplifyBinaryOperation
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:315  (89 body lines, level 3)
//   original  : void ScalarSimplificationsPass::simplifyBinaryOperation( OpBuilder& builder, Operation* op, Type operand_type, PropagationAnalysis& expr_prop_analysis, std::vector<mlir::Operation*>& tobe_deleted)
//   calls     : e240_selectIndicesForUnits, e252_size, e372_areAllExprsValidToTransform, e392_getQueryKeyAndUnitsFromParentRegion, e472_createNewOpOrMap

// crustify:todo: e533_simplifyLoadStoreOperation
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:408  (93 body lines, level 3)
//   original  : void ScalarSimplificationsPass::simplifyLoadStoreOperation( PropagationAnalysis& expr_prop_analysis, OpBuilder& builder, Operation* op)
//   calls     : e240_selectIndicesForUnits, e252_size, e372_areAllExprsValidToTransform, e392_getQueryKeyAndUnitsFromParentRegion, e472_createNewOpOrMap

// crustify:todo: e581_runOnOperation
//   authority : dcc/src/Transform/Sentient/ScalarSimplifications.cpp:872  (54 body lines, level 4)
//   original  : void ScalarSimplificationsPass::runOnOperation()
//   calls     : e471_simplifyConditionals, e532_simplifyBinaryOperation, e533_simplifyLoadStoreOperation

#[cfg(test)]
mod unit_tests {
    use super::{
        ConstraintDim, ConstraintRow, IfCondition, IfPredicate, transform_if_condition,
        update_cmp_i_predicate,
    };
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::CmpPredicate;
    use crate::islands::sentient::dialects::{Op, Val, sentient as ops};

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

    /// One dimension of a constraint row.
    fn dim(value: Val, coeff: i64) -> ConstraintDim {
        ConstraintDim {
            value,
            ty: ScalarTy::Index,
            coeff,
        }
    }

    /// The `sentient.scalar_constant` a transformed side is given.
    fn constant_of(value: i64, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: ops::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// e371 — `%a - %b - 1 >= 0` becomes `%a > %b`, a one-sided row takes a created constant for its
    /// other operand, and a row this `sentient.if` cannot hold changes nothing.
    #[test]
    fn a_constraint_row_becomes_the_ifs_two_operands() {
        let mut values = Values::default();
        let mut consts: Vec<Op> = Vec::new();

        // Both sides present and the constant is the `-1` that turns `sge` into `sgt`.
        let mut op = if_op(CmpPredicate::Sge);
        transform_if_condition(
            IfCondition::of(&mut op).expect("a sentient.if"),
            &ConstraintRow {
                dims: vec![dim(Val(7), 1), dim(Val(8), -1), dim(Val(9), 0)],
                constant: -1,
            },
            &mut consts,
            &mut values,
        );
        assert_eq!(
            op,
            Op::Sentient(ops::Op::If {
                predicate: CmpPredicate::Sgt,
                lhs: Val(7),
                rhs: Val(8),
                yielded: Vec::new(),
                dbg_name: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
            })
        );
        // Nothing was created: neither operand is a constant.
        assert!(consts.is_empty());

        // `%a + 4 >= 0` — the left arg keeps operand 0 and the NEGATED constant becomes operand 1.
        let mut op = if_op(CmpPredicate::Slt);
        transform_if_condition(
            IfCondition::of(&mut op).expect("a sentient.if"),
            &ConstraintRow {
                dims: vec![dim(Val(7), 1)],
                constant: 4,
            },
            &mut consts,
            &mut values,
        );
        let created = Val(0);
        assert_eq!(consts, vec![constant_of(-4, created)]);
        assert_eq!(
            IfCondition::of(&mut op).map(|c| (c.get(), *c.lhs, *c.rhs)),
            Some((CmpPredicate::Sge, Val(7), created))
        );

        // ⛔ THE NO-OP THE FIVE `return`s ARE: a coefficient of 2 is a multiplication this pass does
        // not simplify, so the op and the const builder are both left exactly as they were.
        let mut op = if_op(CmpPredicate::Sge);
        let before = op.clone();
        transform_if_condition(
            IfCondition::of(&mut op).expect("a sentient.if"),
            &ConstraintRow {
                dims: vec![dim(Val(7), 2)],
                constant: -1,
            },
            &mut consts,
            &mut values,
        );
        assert_eq!(op, before);
        assert_eq!(consts.len(), 1);
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
}
