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
