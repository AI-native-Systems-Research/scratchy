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

use crate::islands::sentient::dialects::sentient::CmpPredicate;
use crate::islands::sentient::dialects::{Op, Val, sentient as ops};
use crate::transform::sentient::analyses::PropagationAnalysis;

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
    // `getFirstExprInfo()` on each side — unit index 0, because uniformization supports one DSC and so
    // every unit's conditional is the same (`:207-209`).
    let (minuend, subtrahend, strict) = match predicate {
        CmpPredicate::Sge | CmpPredicate::Eq | CmpPredicate::Ne => (lhs_info, rhs_info, false),
        CmpPredicate::Sle => (rhs_info, lhs_info, false),
        CmpPredicate::Sgt => (lhs_info, rhs_info, true),
        CmpPredicate::Slt => (rhs_info, lhs_info, true),
    };
    let mut all_args: Vec<Val> = expr_prop_analysis.propagated_args(minuend, 0);
    all_args.extend(expr_prop_analysis.propagated_args(subtrahend, 0));
    let equality = matches!(predicate, CmpPredicate::Eq | CmpPredicate::Ne);
    let _ = (strict, equality, all_args, const_scope);
    todo!(
        "affine::getFlattenedAffineExpr and affine::FlatAffineValueConstraints \
         (ScalarSimplifications.cpp:253-306) — out of campaign scope, and with them the local-variable \
         refusal, the redundant-constraint simplification and the one-constraint case that reaches \
         e371_transformIfCondition"
    )
}

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
    use super::{IfPredicate, simplify_conditionals, update_cmp_i_predicate};
    use crate::islands::sentient::dialects::sentient::CmpPredicate;
    use crate::islands::sentient::dialects::{Op, Val, sentient as ops};
    use crate::transform::sentient::analyses::{ExprInfoMap, PropagationAnalysis};

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

        fn affine_expression(&mut self, val: Val) -> ExprInfoMap {
            ExprInfoMap(val.0)
        }

        fn is_expr_info_map_empty(&mut self, _map: ExprInfoMap) -> bool {
            self.empty
        }

        fn propagated_args(&mut self, map: ExprInfoMap, _at: usize) -> Vec<Val> {
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
}
