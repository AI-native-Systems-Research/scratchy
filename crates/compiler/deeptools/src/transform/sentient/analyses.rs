// SPDX-License-Identifier: Apache-2.0
//! THE OUT-OF-SCOPE `Analyses/` SEAM — the handles these passes hold, and never look inside.
//!
//! ⛔⛔ `dcc/src/Transform/Sentient/Analyses/` (39 files) AND `RegisterInitialization/` (12) ARE NOT
//! IN THIS CAMPAIGN — ~11,700 lines, and 122 of the 656 units name one (`crustify-senpass/
//! OUTSIDE-DEPS.tsv` per pass, `OUTSIDE-UNITS.tsv` per unit). What lives here is ONLY the identity
//! of a value such an analysis owns, because a ported field has to have a type.
//!
//! ⛔ EVERY *OPERATION* ON ONE IS A `todo!` NAMING THE ANALYSIS, at the unit that needs it. Do not
//! invent the analysis, do not inline a guess at what it would have returned, and do not substitute
//! a constant for its result.

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Op, Val};

/// AN `EvaluatedValue` THE EXPRESSION EVALUATOR OWNS — an identity, not a value.
///
/// ⛔ 50 OF THE 656 UNITS NAME THIS TYPE and none of them may look inside it:
/// `Analyses/ExpressionEvaluatorUtils.{h,cpp}` is out of campaign scope, so `evaluateSum`,
/// `evaluateMinMax`, `evaluateMultiplyByConst` and `getConstant` are `todo!`s at their call sites.
///
/// ⭐ A HANDLE BECAUSE THE REFERENCE STORES A BORROWED POINTER. Every field that holds one is
/// `const EvaluatedValue *` into the evaluator's own memoisation arena (`AddressPinningAndToggle.cpp`
/// `:174`, `:284`, `:449`, `:527`, `:626`), owned by the one global `ExpressionEvaluator &` the pass
/// threads through — so what a descriptor keeps is which entry, not the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvaluatedValue(pub u32);

/// A SIGNED SCALAR OFFSET — `ScalarValue`, what an evaluated expression carries beside its base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarOffset(pub i64);

/// `BaseValue` — the value an expression is an offset FROM, and whether it is subtracted
/// (`Analyses/ExpressionEvaluatorUtils.h:33-46`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseValue {
    /// `value_`.
    pub value: Val,
    /// `is_negated_` — the base is `-value`, so the expression is `offset - value`.
    pub negated: bool,
}

/// WHAT ONE `evaluateValue` ANSWER TELLS A PORTED PASS — the three queries, and nothing else.
///
/// ⛔⛔ THE THREE FIELDS ARE THE THREE CALLS, NOT A MODEL OF `EvaluatedValue`. The class hierarchy
/// (`AllUnitEvaluatedValue` / the per-unit `offset_vals_` map, `:103-205`) is out of campaign scope,
/// so what is spelled here is exactly `isKnownAbsolute()`, `baseValue()` and
/// `dyn_cast<AllUnitEvaluatedValue>(&v)->offsetValue()` — the only three things
/// `LightweightSimplification.cpp` asks. ⛔ Do not add a per-unit payload by guessing its keys.
///
/// ⭐ `all_unit_offset: None` IS THE FAILED `dyn_cast`, and the distinction is load-bearing:
/// `addOrSubWithZeroSimplification` declines a per-unit value even when it is known absolute
/// (`:58-59`). See [`EvaluatedValue`] for the identity the memoising passes store instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluation {
    /// `isKnownAbsolute()` (`:103`).
    pub known_absolute: bool,
    /// `baseValue()` (`:104`) — `None` when the expression has no base.
    pub base: Option<BaseValue>,
    /// `offsetValue()` (`:163`), and `None` when the value is NOT an `AllUnitEvaluatedValue`.
    pub all_unit_offset: Option<ScalarOffset>,
}

/// WHERE `buildOffsetValue` PUTS THE OPS IT CREATES — the two `OpBuilder`s, as blocks.
///
/// ⛔⛔ THEY ARE TWO DIFFERENT BLOCKS AND ONE OF THEM CAN BE THE BLOCK BEING WALKED.
/// `const_builder` is set to the start of the block the `dataflow.program_unit` *sits in*
/// (`LightweightSimplification.cpp:338-339`), which is never the walked block; `query_map_builder`
/// is set to the start of `getLocalOrGlobalRegion(ops_list.back())`'s front block (`:278-281`),
/// which CAN be it. One `&mut` cannot be handed out twice, so [`OffsetSites::query_maps`] is
/// `None` for exactly that case and the walked block is passed separately.
pub struct OffsetSites<'a> {
    /// `const_builder`'s block — the constants an offset materialises into.
    pub consts: &'a mut Vec<Op>,
    /// `query_map_builder`'s block; `None` says it IS the block being walked.
    pub query_maps: Option<&'a mut Vec<Op>>,
    /// The minter, because every created op binds a fresh value.
    pub values: &'a mut Values,
}

/// THE `ExpressionEvaluator&` A PASS IS HANDED — a trait, because the analysis behind it is not in
/// this campaign and a test must still be able to state its answers.
///
/// ⛔ `Analyses/ExpressionEvaluatorUtils.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only
/// implementation is [`OutOfScopeEvaluator`] and every method of it is a `todo!` naming the analysis.
/// ⭐ `build_offset_value` sits HERE rather than on [`Evaluation`] (where the reference has it, as
/// `EvaluatedValue::buildOffsetValue`, `:126`) so that one out-of-scope seam is one trait.
pub trait ExpressionEvaluator {
    /// `ExpressionEvaluator::evaluateValue` (`Analyses/ExpressionEvaluatorUtils.h:219`).
    fn evaluate_value(&mut self, value: Val) -> Evaluation;

    /// `EvaluatedValue::buildOffsetValue` (`Analyses/ExpressionEvaluatorUtils.h:126`) — materialises
    /// the offset as a value, creating ops in `sites` (`walked` when `sites.query_maps` is `None`).
    fn build_offset_value(
        &mut self,
        evaluation: &Evaluation,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
        ty: ScalarTy,
    ) -> Val;
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeEvaluator;

impl ExpressionEvaluator for OutOfScopeEvaluator {
    fn evaluate_value(&mut self, _value: Val) -> Evaluation {
        todo!(
            "ExpressionEvaluator::evaluateValue (Analyses/ExpressionEvaluatorUtils.h:219) — out of campaign scope"
        )
    }

    fn build_offset_value(
        &mut self,
        _evaluation: &Evaluation,
        _sites: &mut OffsetSites<'_>,
        _walked: &mut Vec<Op>,
        _ty: ScalarTy,
    ) -> Val {
        todo!(
            "EvaluatedValue::buildOffsetValue (Analyses/ExpressionEvaluatorUtils.h:126) — out of campaign scope"
        )
    }
}
