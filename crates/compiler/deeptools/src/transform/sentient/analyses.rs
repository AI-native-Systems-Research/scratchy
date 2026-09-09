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

use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient::RegType;
use crate::islands::sentient::dialects::{Op, Val};
use crate::transform::sentient::canonicalize_xrf_pointers::XrfMinExpr;

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

/// WHAT ONE `evaluateValue` ANSWER TELLS A PORTED PASS — the four queries, and nothing else.
///
/// ⛔⛔ THE FIELDS ARE THE CALLS, NOT A MODEL OF `EvaluatedValue`. The arithmetic and predicates on
/// the class hierarchy (`isAnyValLessThan`, `isDivisibleBy`, `retrieveAllValues`, `:63-205`) are out
/// of campaign scope, so what is spelled here is exactly `isKnownAbsolute()`, `baseValue()`,
/// `AllUnitEvaluatedValue::offsetValue()` and `PerUnitEvaluatedValue::offsetValuesPerUnit()` — every
/// query `LightweightSimplification.cpp` and `ScalarOpMergingAndHoisting.cpp` ask.
///
/// ⭐ A PER-UNIT KIND IS THE FAILED `dyn_cast`, and the distinction is load-bearing:
/// `addOrSubWithZeroSimplification` declines a per-unit value even when it is known absolute
/// (`:58-59`). See [`EvaluatedValue`] for the identity the memoising passes store instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluation {
    /// `isKnownAbsolute()` (`:103`).
    pub known_absolute: bool,
    /// `baseValue()` (`:104`) — `None` when the expression has no base.
    pub base: Option<BaseValue>,
    /// Which of the two `EvaluatedValueKind`s this is, and the offset(s) it carries.
    pub offsets: Offsets,
}

/// WHICH OF THE TWO `EvaluatedValue` KINDS THIS IS, WITH ITS OFFSETS — `enum EvaluatedValueKind
/// { EVK_AllUnit, EVK_PerUnit }` (`Analyses/ExpressionEvaluatorUtils.h:59`).
///
/// ⛔ THE `dyn_cast` PAIR IS THIS MATCH, AND IT IS CLOSED AT TWO. `doesImmutableImmExceedRange`
/// casts to each kind in turn and falls through to `false` for neither
/// (`ScalarOpMergingAndHoisting.cpp:149-164`); with the kind an enum that fall-through is an arm
/// nothing can reach, so the port has none to get wrong.
///
/// ⛔ THE PER-UNIT KEY IS A `Value`, NOT A UNIT INDEX: `typedef llvm::DenseMap<Value, ScalarValue>
/// PerUnitValuesMap` (`:176`), keyed by the `unitKey()` the evaluator built the map against — so
/// the key is stated here from the header's own declaration and not guessed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Offsets {
    /// `AllUnitEvaluatedValue::offsetValue()` (`:163`) — one offset every unit shares.
    AllUnit(ScalarOffset),
    /// `PerUnitEvaluatedValue::offsetValuesPerUnit()` (`:194`) — one offset per unit.
    PerUnit(Vec<(Val, ScalarOffset)>),
}

impl Evaluation {
    /// `dyn_cast<AllUnitEvaluatedValue>(&v)->offsetValue()`, and `None` for the per-unit kind —
    /// the failed cast `addOrSubWithZeroSimplification` turns on
    /// (`LightweightSimplification.cpp:58-59`).
    #[must_use]
    pub fn all_unit_offset(&self) -> Option<ScalarOffset> {
        match &self.offsets {
            Offsets::AllUnit(offset) => Some(*offset),
            Offsets::PerUnit(_) => None,
        }
    }

    /// EVERY OFFSET THIS VALUE CARRIES — the single one of the all-unit kind, or one per unit.
    ///
    /// ⭐ THE TWO `dyn_cast` ARMS OF `doesImmutableImmExceedRange` DIFFER ONLY IN HOW THEY REACH THE
    /// OFFSETS (`ScalarOpMergingAndHoisting.cpp:149-163`): both apply the same range test to every
    /// offset they can see and return on the first failure, so one iterator serves both.
    pub fn offset_values(&self) -> impl Iterator<Item = ScalarOffset> + '_ {
        let (all_unit, per_unit): (Option<ScalarOffset>, &[(Val, ScalarOffset)]) =
            match &self.offsets {
                Offsets::AllUnit(offset) => (Some(*offset), &[]),
                Offsets::PerUnit(offsets) => (None, offsets.as_slice()),
            };
        all_unit
            .into_iter()
            .chain(per_unit.iter().map(|&(_, offset)| offset))
    }
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

    /// `ExpressionEvaluator::evaluateSum` (`Analyses/ExpressionEvaluatorUtils.h:242`) — the
    /// evaluation of `lhs + rhs`, which is how a hoist folds an increment into a constant it already
    /// evaluated.
    fn evaluate_sum(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation;

    /// `ExpressionEvaluator::evaluateSub` (`Analyses/ExpressionEvaluatorUtils.h:260`) — the
    /// evaluation of `lhs - rhs`, which is how a reordered toggle turns `minuend - init` into the
    /// loop's new initializer (`ToggleReordering.cpp:128-129`).
    ///
    /// ⭐ DEFAULTED, like [`ExpressionEvaluator::build_offset_value_of`]: the out-of-scope refusal is
    /// stated once, and a test double for a pass that only ever sums need not repeat it.
    fn evaluate_sub(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation {
        let _ = (lhs, rhs);
        todo!(
            "ExpressionEvaluator::evaluateSub (Analyses/ExpressionEvaluatorUtils.h:260) — out of campaign scope"
        )
    }

    /// `EvaluatedValue::buildOffsetValue` (`Analyses/ExpressionEvaluatorUtils.h:126`) — materialises
    /// the offset as a value, creating ops in `sites` (`walked` when `sites.query_maps` is `None`).
    fn build_offset_value(
        &mut self,
        evaluation: &Evaluation,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
        ty: ScalarTy,
    ) -> Val;

    /// The SAME `EvaluatedValue::buildOffsetValue` (`:126`) reached from a STORED handle rather than a
    /// fresh [`Evaluation`] — what a memoising pass has when it comes back to build the value.
    ///
    /// ⛔ THE HANDLE IS NOT AN [`Evaluation`] AND CANNOT BE TURNED INTO ONE HERE: the arena entry it
    /// names is the analysis's, so `ScalarOpMerging::unrollBurstAndIL` (`:1042`) calls the method on
    /// `const EvaluatedValue *` directly. ⭐ DEFAULTED so the out-of-scope refusal is stated once and
    /// a test double that never unrolls need not repeat it.
    fn build_offset_value_of(
        &mut self,
        immutable: EvaluatedValue,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
        ty: ScalarTy,
    ) -> Val {
        let _ = (immutable, sites, walked, ty);
        todo!(
            "EvaluatedValue::buildOffsetValue (Analyses/ExpressionEvaluatorUtils.h:126) — out of campaign scope"
        )
    }

    // ─────────────────────────── the HANDLE flavour ───────────────────────────
    //
    // ⛔⛔ THE SAME C++ FUNCTIONS, ASKED FOR THE ARENA ENTRY RATHER THAN ITS CONTENTS, and the two
    // flavours are not interchangeable. `evaluateValue` and `evaluateSum` return
    // `const EvaluatedValue &` (`Analyses/ExpressionEvaluatorUtils.h:219`, `:242`); a pass that
    // inspects offsets decodes that into an [`Evaluation`], but every one of the six
    // `AddressPinningAndToggle` descriptors STORES the reference itself (`:198`, `:285`, `:339`,
    // `:432`, `:528`, `:628`) and compares stored handles with `==`. Decoding for them would be inventing the
    // analysis; see [`EvaluatedValue`].
    // ⭐ ALL DEFAULTED, like [`ExpressionEvaluator::evaluate_sub`]: the out-of-scope refusal is
    // stated once here and a test double for a pass that never asks need not repeat it.

    /// `ExpressionEvaluator::evaluateValue` (`Analyses/ExpressionEvaluatorUtils.h:219`) for its
    /// HANDLE — what a descriptor keeps.
    fn evaluate_value_handle(&mut self, value: Val) -> EvaluatedValue {
        let _ = value;
        todo!(
            "ExpressionEvaluator::evaluateValue (Analyses/ExpressionEvaluatorUtils.h:219) — out of campaign scope"
        )
    }

    /// `ExpressionEvaluator::getConstant` (`Analyses/ExpressionEvaluatorUtils.h:335`) — the evaluated
    /// value of a literal, which is how a descriptor obtains the `zero_ev` it tests against.
    fn constant(&mut self, value: i64) -> EvaluatedValue {
        let _ = value;
        todo!(
            "ExpressionEvaluator::getConstant (Analyses/ExpressionEvaluatorUtils.h:335) — out of campaign scope"
        )
    }

    /// `ExpressionEvaluator::evaluateSum` (`Analyses/ExpressionEvaluatorUtils.h:242`) on two handles.
    fn evaluate_sum_handle(&mut self, lhs: EvaluatedValue, rhs: EvaluatedValue) -> EvaluatedValue {
        let _ = (lhs, rhs);
        todo!(
            "ExpressionEvaluator::evaluateSum (Analyses/ExpressionEvaluatorUtils.h:242) — out of campaign scope"
        )
    }

    /// `ExpressionEvaluator::evaluateSub` (`Analyses/ExpressionEvaluatorUtils.h:260`) on two handles
    /// — how a conditional-constant updater rebases each yielded constant on the new immutable
    /// address (`AddressPinningAndToggle.cpp:2073`).
    fn evaluate_sub_handle(&mut self, lhs: EvaluatedValue, rhs: EvaluatedValue) -> EvaluatedValue {
        let _ = (lhs, rhs);
        todo!(
            "ExpressionEvaluator::evaluateSub (Analyses/ExpressionEvaluatorUtils.h:260) — out of campaign scope"
        )
    }

    /// `EvaluatedValue::operator==` (`Analyses/ExpressionEvaluatorUtils.h:63`) — whether two arena
    /// entries hold the SAME value, which is not the handle identity `==` on [`EvaluatedValue`] gives:
    /// `*val == *(*it_next)->getLB()` (`CFGSimplificationSentientLevel.cpp:438`) asks about a value
    /// the evaluator has just summed, and only the analysis can compare the two.
    fn values_equal(&mut self, lhs: EvaluatedValue, rhs: EvaluatedValue) -> bool {
        let _ = (lhs, rhs);
        todo!(
            "EvaluatedValue::operator== (Analyses/ExpressionEvaluatorUtils.h:63) — out of campaign scope"
        )
    }

    /// `ExpressionEvaluator::evaluateMultiplyByConst`
    /// (`Analyses/ExpressionEvaluatorUtils.h:285`) — `ev * by`.
    fn evaluate_multiply_by_const(&mut self, ev: EvaluatedValue, by: i64) -> EvaluatedValue {
        let _ = (ev, by);
        todo!(
            "ExpressionEvaluator::evaluateMultiplyByConst (Analyses/ExpressionEvaluatorUtils.h:285) — out of campaign scope"
        )
    }

    /// `ExpressionEvaluator::evaluateMinMax` (`Analyses/ExpressionEvaluatorUtils.h:318`) — the
    /// element-wise minimum or maximum of a list.
    fn evaluate_min_max(&mut self, values: &[EvaluatedValue], which: MinMax) -> EvaluatedValue {
        let _ = (values, which);
        todo!(
            "ExpressionEvaluator::evaluateMinMax (Analyses/ExpressionEvaluatorUtils.h:318) — out of campaign scope"
        )
    }

    /// `EvaluatedValue::isAnyValLessThan` (`Analyses/ExpressionEvaluatorUtils.h:71`) — whether ANY
    /// unit's value is below `bound`, which is how a chain refuses a negative stride.
    fn is_any_val_less_than(&mut self, ev: EvaluatedValue, bound: ScalarOffset) -> bool {
        let _ = (ev, bound);
        todo!(
            "EvaluatedValue::isAnyValLessThan (Analyses/ExpressionEvaluatorUtils.h:71) — out of campaign scope"
        )
    }

    /// THE FOUR QUERIES OF [`Evaluation`] ASKED OF A STORED HANDLE — `isKnownAbsolute()`, `baseValue()`
    /// and the offsets (`Analyses/ExpressionEvaluatorUtils.h:103-104`, `:118`, `:141`) read off the
    /// arena entry `ev` names, exactly as [`ExpressionEvaluator::evaluate_value`] reads them off the
    /// entry it has just made.
    ///
    /// ⛔ THE DECODE IS THE ANALYSIS'S, NOT OURS — see the note on
    /// [`ExpressionEvaluator::build_offset_value_of`]: nothing here may reconstruct an [`Evaluation`]
    /// from a handle. It is a seam because a pass that MEMOISES still has to ask a ported range test a
    /// question about what it stored: `ScalarOpMerging::isFieldUnrollCandidate` (`:1160`) hands
    /// `doesImmutableImmExceedRange` a `const EvaluatedValue &` it has been carrying for two loops.
    fn evaluation_of(&mut self, ev: EvaluatedValue) -> Evaluation {
        let _ = ev;
        todo!(
            "EvaluatedValue::isKnownAbsolute/baseValue/offsetValue (Analyses/ExpressionEvaluatorUtils.h:103) — out of campaign scope"
        )
    }
}

/// WHICH END OF `evaluateMinMax` IS WANTED — the reference's `bool compute_min`
/// (`Analyses/ExpressionEvaluatorUtils.h:318`), which its two adjacent call sites pass `false` then
/// `true` (`AddressPinningAndToggle.cpp:3043-3046`) and which a `bool` argument cannot keep apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinMax {
    /// `compute_min = true`.
    Min,
    /// `compute_min = false`.
    Max,
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

    fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
        todo!(
            "ExpressionEvaluator::evaluateSum (Analyses/ExpressionEvaluatorUtils.h:242) — out of campaign scope"
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

/// WHICH REGION OF WHICH REGION-OWNING OP A DATA TRANSFER SITS IN — the reference's
/// `std::pair<mlir::Operation *, int> region_op_and_region_num` (`AddressPinningAndToggle.cpp:1937`),
/// the key a DYNAMIC pinning scheme looks the unit up by when an address is unit-independent
/// (`Analyses/AddressPinningScheme.h:196-198`, `:225-227`).
///
/// ⛔ AN ENUM BECAUSE ONLY TWO OPS EVER OWN ONE HERE, and `collectDataTransfers` is where both are
/// spelled: the `dataflow.program_unit` itself with region `0` (`:1319`) and a
/// `uniform.uniformize_regions` with the region's own index (`:1306`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegionSite {
    /// `collectDataTransfers(comp, op, memory_unit, unit, 0)` (`:1319`) — the unit's own body.
    ///
    /// ⭐ THE DEFAULT, because a descriptor built without one describes a transfer the walk found
    /// directly in the unit (`:1319` is the walk's OWN fall-through, `:1306` its uniformized branch).
    #[default]
    ProgramUnitBody,
    /// `collectDataTransfers(comp, oper, memory_unit, uniform_op, i)` (`:1306`) — region `i` of the
    /// `uniform.uniformize_regions` at position `at` of the unit body, position being this island's op
    /// identity (see [`OpId`]).
    UniformizedRegion {
        /// Where that `uniform.uniformize_regions` sits.
        at: BodyIndex,
        /// Which of its regions — `i`, the walk's own region index.
        region: RegionNum,
    },
}

/// A POSITION IN A UNIT BODY — `uniform_op`, named the way this island names an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyIndex(pub usize);

/// WHICH REGION OF ITS OWNER — the `int` half of `region_op_and_region_num`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionNum(pub usize);

/// THE `const PinningSchemeManager&` A PASS IS HANDED — a trait, for the reason
/// [`ExpressionEvaluator`] is one: `Analyses/AddressPinningScheme.{h,cpp}` is not in this campaign and
/// a test must still be able to state which pinned address it chose.
pub trait PinningSchemeManager {
    /// `PinningSchemeManager::findClosestPinnedAddr(ev_x, ev_y, region_op_and_region_num,
    /// element_size_in_bits)` (`Analyses/AddressPinningScheme.h:229-233`) — the closest pinned
    /// address, in ELEMENT addresses, for a pair of toggling addresses.
    ///
    /// ⭐ THE ONE-ADDRESS OVERLOAD (`:208-219`) IS THIS SAME CALL WITH `X == Y`, by its own body, and
    /// it is out-of-scope code rather than a campaign unit — so a caller with one address passes it
    /// twice instead of there being a second method here.
    fn find_closest_pinned_addr(
        &self,
        ev_x: EvaluatedValue,
        ev_y: EvaluatedValue,
        region: RegionSite,
        element_size: Bits,
    ) -> EvaluatedValue {
        let _ = (ev_x, ev_y, region, element_size);
        todo!(
            "PinningSchemeManager::findClosestPinnedAddr (Analyses/AddressPinningScheme.h:229) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION, for the reason [`OutOfScopeEvaluator`] is the evaluator's.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopePinningSchemeManager;

impl PinningSchemeManager for OutOfScopePinningSchemeManager {}

/// THE `XRFRegisterAnalyzer *` A PT PASS IS HANDED — a trait for the same reason
/// [`ExpressionEvaluator`] is one: the analyzer is not in this campaign, and a test must still be
/// able to state its answers.
///
/// ⛔ `Analyses/XRFRegisterAnalyzer.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only
/// implementation is [`OutOfScopeXrfRegisterAnalyzer`] and every method of it is a `todo!`.
pub trait XrfRegisterAnalyzer {
    /// `getMinMaxValIfConstant(value, min)` (`Analyses/XRFRegisterAnalyzer.h:97`) — the smallest or
    /// largest value `value` is known to take, `None` when it is not a constant range.
    fn min_max_val_if_constant(&mut self, value: Val, end: MinMax) -> Option<i64> {
        let _ = (value, end);
        todo!(
            "XRFRegisterAnalyzer::getMinMaxValIfConstant (Analyses/XRFRegisterAnalyzer.h:97) — out of campaign scope"
        )
    }

    /// `XRFRegisterAnalyzer(unit)` then `getValToMinExprMap()` — every xrf-related value whose
    /// minimal expression the analyzer computed for THIS unit, each with `getValIfConstant`'s answer.
    ///
    /// ⭐ THE PAIR IS [`XrfMinExpr`], which `replace_const_xrf_expressions` already declared for the
    /// same reason: the analyzer is out of scope, so both of its answers arrive as data.
    fn val_to_min_expr(&mut self, unit: &[Op]) -> Vec<XrfMinExpr> {
        let _ = unit;
        todo!(
            "XRFRegisterAnalyzer::getValToMinExprMap (Analyses/XRFRegisterAnalyzer.h:76) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION: the analyzer is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeXrfRegisterAnalyzer;

impl XrfRegisterAnalyzer for OutOfScopeXrfRegisterAnalyzer {}

/// AN INSTRUCTION COUNT FROM `InstructionEstimatorImpl` — the reference's `int`.
///
/// ⛔ SIGNED BECAUSE THE REFERENCE'S IS: `getRemainingIbuffSpace` goes negative once a unit overruns
/// the buffer, and the split/unroll cost model compares against it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionCount(pub i32);

/// THE `InstructionEstimatorImpl&` A PASS IS HANDED — a trait for the same reason
/// [`ExpressionEvaluator`] is one: the estimator is not in this campaign, and a test must still be
/// able to state its answers.
///
/// ⛔ `Analyses/InstructionEstimation.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only
/// implementation is [`OutOfScopeInstructionEstimator`] and every method of it is a `todo!`.
/// ⭐ THE REFERENCE OVERLOADS ON THE ARGUMENT KIND (op, block, region, unit); the two overloads the
/// ported units call are named apart here because Rust has no overloading.
pub trait InstructionEstimator {
    /// `recalculate(ctx, unit)` (`Analyses/InstructionEstimation.h:74`) — recounts the whole unit.
    fn recalculate(&mut self, unit: &[Op]);

    /// `getEstimatedInstructionCount(ctx, Operation *)` (`Analyses/InstructionEstimation.h:56`) —
    /// the op and everything nested in it.
    fn estimated_instruction_count_of_op(&mut self, op: &Op) -> InstructionCount;

    /// `getEstimatedInstructionCount(ctx, Region *)` (`Analyses/InstructionEstimation.h:62`).
    fn estimated_instruction_count_of_region(&mut self, region: &[Op]) -> InstructionCount;
}

/// THE ONE CRATE IMPLEMENTATION: the estimator is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeInstructionEstimator;

impl InstructionEstimator for OutOfScopeInstructionEstimator {
    fn recalculate(&mut self, _unit: &[Op]) {
        todo!(
            "InstructionEstimatorImpl::recalculate (Analyses/InstructionEstimation.h:74) — out of campaign scope"
        )
    }

    fn estimated_instruction_count_of_op(&mut self, _op: &Op) -> InstructionCount {
        todo!(
            "InstructionEstimatorImpl::getEstimatedInstructionCount (Analyses/InstructionEstimation.h:56) — out of campaign scope"
        )
    }

    fn estimated_instruction_count_of_region(&mut self, _region: &[Op]) -> InstructionCount {
        todo!(
            "InstructionEstimatorImpl::getEstimatedInstructionCount (Analyses/InstructionEstimation.h:62) — out of campaign scope"
        )
    }
}

/// A `Candidate` THE REGISTER-INITIALISATION PIPELINE OWNS — an identity, not a candidate.
///
/// ⛔ `RegisterInitialization/Candidate.{h,cpp}` IS OUT OF CAMPAIGN SCOPE. The reference's lists are
/// `SmallVector<Candidate *>` (`Candidate.h:65`) whose entries the collector `delete`s in its own
/// destructor (`RegisterInitialization/Collector.h:59-62`), so what a ported list holds is WHICH
/// candidate, not the candidate — and nothing here may look inside one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Candidate(pub u32);

/// THE `CollectorInterface &` THE `Driver` IS HANDED (`RegisterInitialization/Collector.h:29`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE, so the crate's only implementation is [`OutOfScopeCandidateCollector`].
/// ⭐ `&mut dyn` AT THE CALL SITES BECAUSE THE REFERENCE PICKS THE IMPLEMENTATION AT RUN TIME, from
/// `llvm::cl::opt<CollectorKind>` (`RegisterInitialization.cpp:40-52`).
pub trait CandidateCollector {
    /// `collectGlobalCandidates(results)` — the reference expects `results` empty on entry.
    fn collect_global_candidates(&mut self, results: &mut Vec<Candidate>);

    /// `collectLocalCandidates(core, results)` — `core` is a group leader's unit value.
    fn collect_local_candidates(&mut self, core: Val, results: &mut Vec<Candidate>);
}

/// THE `EvaluatorInterface &` THE `Driver` IS HANDED (`RegisterInitialization/Evaluator.h:36`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeCandidateEvaluator`]. Only the two methods
/// `runLocalAnalysis` calls are declared; `evaluateGlobally` lands with e346.
pub trait CandidateEvaluator {
    /// `evaluateLocally(candidates)` — weights, sorts and re-prioritises IN PLACE.
    fn evaluate_locally(&mut self, candidates: &mut Vec<Candidate>);

    /// `mergeInto(result, sublist)` (`Evaluator.h:66`) — inserts `sublist` keeping `result` sorted.
    fn merge_into(&mut self, result: &mut Vec<Candidate>, sublist: &[Candidate]);
}

/// THE `SelectorInterface &` THE `Driver` IS HANDED (`RegisterInitialization/Selector.h:38`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeCandidateSelector`]. `selectGlobally` lands with e346.
pub trait CandidateSelector {
    /// `selectLocally(local, global, core)` (`Selector.h:68`) — PURGES both lists in place; neither
    /// can grow.
    fn select_locally(
        &mut self,
        local: &mut Vec<Candidate>,
        global: &mut Vec<Candidate>,
        core: Val,
    );
}

/// THE `const UniformGroupAnalyzer &` THE `Driver` IS HANDED
/// (`Analyses/UniformGroupAnalysis.h:57`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeUniformGroups`]. Only `getGroupLeaders` is declared.
pub trait UniformGroups {
    /// `getGroupLeaders()` (`Analyses/UniformGroupAnalysis.h:67`) — one unit value per exclusive
    /// group.
    fn group_leaders(&self) -> Vec<Val>;

    /// `isGroupLeader(unit)` (`Analyses/UniformGroupAnalysis.h:68-71`).
    ///
    /// ⛔ A KEY TEST ON `non_leader_group_members_`, NOT A FOLLOWER COUNT: a leader whose group has no
    /// followers still answers `true`.
    fn is_group_leader(&self, unit: Val) -> bool;

    /// `getGroupMembersLedBy(leader)` (`Analyses/UniformGroupAnalysis.h:72-76`) — the followers, not
    /// including the leader.
    fn group_members_led_by(&self, leader: Val) -> Vec<Val>;
}

/// THE ONE CRATE IMPLEMENTATION of [`CandidateCollector`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeCandidateCollector;

impl CandidateCollector for OutOfScopeCandidateCollector {
    fn collect_global_candidates(&mut self, _results: &mut Vec<Candidate>) {
        todo!(
            "CollectorInterface::collectGlobalCandidates (RegisterInitialization/Collector.h:37) — out of campaign scope"
        )
    }

    fn collect_local_candidates(&mut self, _core: Val, _results: &mut Vec<Candidate>) {
        todo!(
            "CollectorInterface::collectLocalCandidates (RegisterInitialization/Collector.h:47) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`CandidateEvaluator`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeCandidateEvaluator;

impl CandidateEvaluator for OutOfScopeCandidateEvaluator {
    fn evaluate_locally(&mut self, _candidates: &mut Vec<Candidate>) {
        todo!(
            "EvaluatorInterface::evaluateLocally (RegisterInitialization/Evaluator.h:48) — out of campaign scope"
        )
    }

    fn merge_into(&mut self, _result: &mut Vec<Candidate>, _sublist: &[Candidate]) {
        todo!(
            "EvaluatorInterface::mergeInto (RegisterInitialization/Evaluator.h:66) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`CandidateSelector`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeCandidateSelector;

impl CandidateSelector for OutOfScopeCandidateSelector {
    fn select_locally(
        &mut self,
        _local: &mut Vec<Candidate>,
        _global: &mut Vec<Candidate>,
        _core: Val,
    ) {
        todo!(
            "SelectorInterface::selectLocally (RegisterInitialization/Selector.h:68) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`UniformGroups`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeUniformGroups;

impl UniformGroups for OutOfScopeUniformGroups {
    fn group_leaders(&self) -> Vec<Val> {
        todo!(
            "UniformGroupAnalyzer::getGroupLeaders (Analyses/UniformGroupAnalysis.h:67) — out of campaign scope"
        )
    }

    fn is_group_leader(&self, _unit: Val) -> bool {
        todo!(
            "UniformGroupAnalyzer::isGroupLeader (Analyses/UniformGroupAnalysis.h:68) — out of campaign scope"
        )
    }

    fn group_members_led_by(&self, _leader: Val) -> Vec<Val> {
        todo!(
            "UniformGroupAnalyzer::getGroupMembersLedBy (Analyses/UniformGroupAnalysis.h:72) — out of campaign scope"
        )
    }
}

/// A `LiveRange` THE PORT-ASSIGNMENT GRAPH KEEPS PER OPERAND — an identity, not the intervals.
///
/// ⛔ `Analyses/LiveRange.{hpp,cpp}` IS OUT OF CAMPAIGN SCOPE, so the `std::vector<LabeledRange>`
/// behind it is deliberately absent and `overlaps`/`unionWith` are `todo!`s at the units that need
/// them — `e572_computePortLiveRange` builds these and `e608_buildGraphEdges` reads them.
/// `e123_clean`, which only empties the map holding them, needs the type and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveRange;

/// WHERE A LIVE RANGE OPENS — `getIntervals()[0].first`, the instruction index its first labelled
/// interval starts at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RangeStart(pub u32);

impl LiveRange {
    /// `getIntervals()[0].first` (`PortAssignment.cpp:751`).
    ///
    /// ⛔ `Analyses/LiveRange.{hpp,cpp}` IS OUT OF CAMPAIGN SCOPE, so this is where
    /// `e341_addReuseToDummyOperands` stops — everything up to and including the owners-differ
    /// short-circuit is ported, and the interval comparison is not.
    #[must_use]
    pub fn first_interval_start(self) -> RangeStart {
        todo!("LiveRange::getIntervals (Analyses/LiveRange.hpp:44) — out of campaign scope")
    }
}

/// A NODE OF A COLOURING GRAPH — `GraphNode::index_`, which for port assignment is the operand's data
/// id (`Analyses/GraphColoring.cpp:58`).
///
/// ⛔ THERE IS NO NODE FOR THE REFERENCE'S `-1`: `getOrAddNode(-1)` answers `nullptr` and its callers
/// dereference it, so an unassigned data id is a crash there (`GraphColoring.cpp:61`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphNodeId(pub u32);

/// ONE COLOUR A NODE MAY TAKE — an element of `GraphNode::possible_values_`, which port assignment
/// colours with the three compute ports (`Analyses/GraphColoring.hpp:41`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphColor(pub u32);

/// THE `GraphColoring` A PASS OWNS — a trait, for the same reason [`ExpressionEvaluator`] is one: the
/// analysis is not in this campaign and a test must still be able to observe what a pass asks it.
///
/// ⛔ `Analyses/GraphColoring.{hpp,cpp}` IS OUT OF CAMPAIGN SCOPE. Only the methods a ported unit
/// actually calls are declared; `getOrAddNode`, `addBidirectionalEdge`, `addSameColorEdge` and
/// `doGraphColoring` belong to `e339`, `e608` and `e382` and are added by those units.
pub trait ColoringGraph {
    /// `GraphColoring::clear()` (`Analyses/GraphColoring.hpp:52-60`).
    ///
    /// ⛔⛔ NOT A RE-DEFAULT-CONSTRUCTION, AND THE DIFFERENCE OUTLIVES A PROGRAM UNIT. It deletes and
    /// clears `nodes_`, then clears `node_ids_` and `edges_` — and leaves `same_color_edges_` and
    /// `max_node_id_` STANDING, so both survive into the next unit the pass visits even though the
    /// constructor's own `clear()` ran on an empty object. Whoever gives this an interior must never
    /// write `*self = Self::default()`.
    fn clear(&mut self);

    /// `GraphColoring::getOrAddNode(int)` (`Analyses/GraphColoring.cpp:58-68`).
    ///
    /// ⭐ CREATING THE NODE IS THE WHOLE EFFECT — the `GraphNode *` it hands back is the mechanism for
    /// reaching it, which is why the two questions asked of a node are keyed by id here instead.
    fn get_or_add_node(&mut self, node: GraphNodeId);

    /// `GraphNode::addValidValues(v)` at its default `addon = false`
    /// (`Analyses/GraphColoring.cpp:45-56`).
    ///
    /// ⛔⛔ IT **REPLACES** THE NODE'S LIST, IT DOES NOT ADD TO IT. Every call in
    /// `e339_addNodesToGraph` omits `addon`, so the last one to name a node wins outright — which is
    /// what makes the trivial-FMA and `DataTransferOnly` calls at the end of that unit overrides
    /// rather than intersections.
    fn add_valid_values(&mut self, node: GraphNodeId, values: &[GraphColor]);

    /// `GraphNode::getValidValues()` (`Analyses/GraphColoring.hpp:37`) — empty for a node nothing has
    /// constrained, which is `possible_values_`'s own initial state.
    fn valid_values(&self, node: GraphNodeId) -> Vec<GraphColor>;
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeColoringGraph;

impl ColoringGraph for OutOfScopeColoringGraph {
    fn clear(&mut self) {
        todo!("GraphColoring::clear (Analyses/GraphColoring.hpp:52) — out of campaign scope")
    }

    fn get_or_add_node(&mut self, node: GraphNodeId) {
        todo!(
            "GraphColoring::getOrAddNode({node:?}) (Analyses/GraphColoring.cpp:58) — out of \
             campaign scope"
        )
    }

    fn add_valid_values(&mut self, node: GraphNodeId, values: &[GraphColor]) {
        todo!(
            "GraphNode::addValidValues({node:?}, {values:?}) (Analyses/GraphColoring.cpp:45) — out \
             of campaign scope"
        )
    }

    fn valid_values(&self, node: GraphNodeId) -> Vec<GraphColor> {
        todo!(
            "GraphNode::getValidValues({node:?}) (Analyses/GraphColoring.hpp:37) — out of campaign \
             scope"
        )
    }
}

/// `RegisterGraphs` (`Analyses/GraphColoring.hpp:152`) — one [`ColoringGraph`] per register locale,
/// plus their hyper-graphs and the value-to-locale cache, as the allocator holds it.
pub trait RegisterGraphs {
    /// `RegisterGraphs::clean()` (`Analyses/GraphColoring.hpp:179-183`).
    ///
    /// ⛔⛔ IT LEAVES `ec_map_` STANDING. Three of the four members are cleared and the same-colour
    /// equivalence classes are not, so they survive into the next program unit the allocator visits —
    /// the same shape of trap as [`ColoringGraph::clear`], and for the same reason nobody giving this
    /// an interior may write `*self = Self::default()`.
    fn clean(&mut self);
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeRegisterGraphs;

impl RegisterGraphs for OutOfScopeRegisterGraphs {
    fn clean(&mut self) {
        todo!("RegisterGraphs::clean (Analyses/GraphColoring.hpp:179) — out of campaign scope")
    }
}

/// `RDENode` (`Analyses/RedundantDefinitionEliminationTree.hpp:34`) AS THE PORTED PASSES READ IT —
/// the tree itself is out of campaign scope, so only the two facts an `initializeDataflowInfo` or an
/// `isSimplifiable` asks of a node are represented.
///
/// ⭐ `Root` IS THE IDENTITY TEST, NOT A FLAG: `root_ = root_ ? root_ : new RDENode(nullptr)`
/// (`Analyses/RedundantDefinitionEliminationTree.cpp:223`) makes the root the ONLY node without an
/// operation, so `getRoot() == &node` is a CASE of this enum rather than a pointer comparison.
///
/// ⭐ ONE DEFINITION FOR THE WHOLE CAMPAIGN. Four RDE passes (`ImplicitSyncRE`,
/// `SetActiveMaskValueRE`, `SetMaskRE`, `SetSendDestinationRE`) override the same two hooks and ask a
/// node the same two questions, so this lives at the out-of-scope seam and not in one pass's module.
#[derive(Debug, Clone, Copy)]
pub enum RdeNode<'a> {
    /// The tree's root — `getOperation()` is null.
    Root,
    /// A node over one op.
    At {
        /// `getOperation()`.
        op: &'a Op,
        /// `isLeaf()` (`src/Analysis/OperationTree.hpp:70`).
        leaf: bool,
    },
}

/// THE `Liveness&` A PASS IS HANDED — a trait for the same reason [`ExpressionEvaluator`] is one:
/// the analysis is not in this campaign, and a test must still be able to observe WHICH values a
/// ported pass promotes.
///
/// ⛔ `Analyses/Liveness.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only implementation is
/// [`OutOfScopeLiveness`] and every method of it is a `todo!`.
pub trait Liveness {
    /// `updateLiveRangesForProgramHeaderPromotion(candidate)` (`Analyses/Liveness.h:130`) — widens
    /// `candidate`'s live range to the whole program because it is about to live in the header.
    fn update_live_ranges_for_program_header_promotion(&mut self, candidate: Val);

    /// `isLiveRangeOverlaps(val1, val2)` (`Analyses/Liveness.h:111`) — whether the two values are ever
    /// live at the same time, i.e. whether they may NOT share a register.
    fn is_live_range_overlaps(&self, val1: Val, val2: Val) -> bool;
}

/// THE ONE CRATE IMPLEMENTATION: liveness is not ported, so telling it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeLiveness;

impl Liveness for OutOfScopeLiveness {
    fn update_live_ranges_for_program_header_promotion(&mut self, _candidate: Val) {
        todo!(
            "Liveness::updateLiveRangesForProgramHeaderPromotion (Analyses/Liveness.h:130) — out of campaign scope"
        )
    }

    fn is_live_range_overlaps(&self, _val1: Val, _val2: Val) -> bool {
        todo!("Liveness::isLiveRangeOverlaps (Analyses/Liveness.h:111) — out of campaign scope")
    }
}

/// A COUNT OF CYCLES — what `TimeStamp::getCyclesGap` answers and what an exposed-pipeline budget is
/// spent in (`Analyses/TimeStamps.h:20`, `TransformForExposedPipeline.cpp:308`).
///
/// ⛔ SIGNED, AND NEGATIVE IS A CASE THE CALLERS READ: `computeDependenciesSameBlock` drops a pair on
/// `gap < 0` (`:164`), and `insertNOPOperations` lets its own budget go below zero and still banks it
/// (`:333-337`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Cycles(pub i32);

/// ONE COLUMN OF A `TimeStamp` — `TimeStampColumnVal` (`Analyses/TimeStamps.h:40`) AS A PORTED PASS
/// HOLDS IT: an identity, like [`LiveRange`], because the loop or condition it names, and the
/// iteration count beside it, belong to the analysis.
///
/// ⛔ `Analyses/TimeStamps.{h,cpp}` IS OUT OF CAMPAIGN SCOPE. `isLoop`, `isCond`, `getLoop` and
/// `getCond` are `todo!`s at the units that need them — `e482_computeDependenciesSameBlock` walks a
/// timestamp to find a parent op, and `e234_printDependencies` only carries the vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimeStampColumnVal;

/// `Dependency` (`Analyses/TimeStamps.h:30-34`) — one RAW hazard between two MACs, and how many
/// cycles apart they are.
///
/// ⛔ THE ENDS ARE POSITIONS, NOT `Operation *`: [`OpId`] is this crate's stand-in, so every entry in
/// a list of these GOES STALE the moment an op is inserted before it — which is exactly what
/// `e237_insertNOPOperations` does, and why it computes every count before it moves anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    /// `src` — the MAC whose result is written.
    pub src: OpId,
    /// `dst` — the MAC that reads it.
    pub dst: OpId,
    /// `gap`.
    pub gap: Cycles,
}

/// WHICH UNIT AN INDEX-KEYED ANALYSIS RESULT BELONGS TO — a position in the value range of
/// `PropagationAnalysis::getUnitIndexMap()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnitIndex(pub u32);

/// THE `std::unordered_map<std::string, unsigned>` `selectIndicesForUnits` IS HANDED — a trait,
/// because BOTH halves of the lookup are out of campaign scope.
///
/// ⛔ THE MAP IS `PropagationAnalysis::getUnitIndexMap()` at every call site
/// (`ScalarSimplifications.cpp:335`, `:420`, `:554`, `Analyses/PropagationAnalysis.cpp:954`) and the
/// key is `getFoldedUnitNameAsString` (`Analyses/Utils.cpp:644`); `Analyses/` is not in this
/// campaign, so the crate's only implementation is [`OutOfScopeUnitIndexMap`].
///
/// ⭐ KEYED BY [`Val`], NOT BY A STRING. The name is a formatter over the unit op's own attributes,
/// and it is the map's key only because C++ has no hashable `Value` for an `unordered_map`.
pub trait UnitIndexMap {
    /// `unit_name_to_index_map.at(getFoldedUnitNameAsString(unit))` (`Utils.cpp:78-79`).
    fn index_of(&self, unit: Val) -> UnitIndex;
}

/// THE ONE CRATE IMPLEMENTATION: neither the map nor the name formatter behind it is ported.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeUnitIndexMap;

impl UnitIndexMap for OutOfScopeUnitIndexMap {
    fn index_of(&self, _unit: Val) -> UnitIndex {
        todo!(
            "PropagationAnalysis::getUnitIndexMap + getFoldedUnitNameAsString \
             (Analyses/PropagationAnalysis.h:325, Analyses/Utils.cpp:644) — out of campaign scope"
        )
    }
}

/// `AffineMap` AS THE PORTED PASSES USE ONE — an identity, plus the single structural fact
/// `getBaseExpr` reads off it.
///
/// ⛔ MLIR UPSTREAM AND OUT OF CAMPAIGN SCOPE, exactly like `affine::FlatAffineValueConstraints`: the
/// map is built by `PropagationAnalysis` and flattened by `mlir::getFlattenedAffineExpr`, neither of
/// which is in this scope. `getResult(0)` is the only expression ever taken from it
/// (`LiveRangeReduction.cpp:308`), so the map does not have to be indexable here.
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
/// ⛔ NOT `LiveRangeReductionPass::ExprInfo`, a different class of the same name
/// (`LiveRangeReduction.cpp:116`) that lives with the pass that owns it.
#[derive(Debug, Clone, Default)]
pub struct PropagatedExpr {
    /// `propagated_map_`.
    pub propagated_map: PropagatedMap,
    /// `propagated_args_` — the SSA values the map's dimensions stand for.
    pub propagated_args: Vec<Val>,
    /// `cannot_be_resolved_`.
    pub cannot_be_resolved: bool,
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

    /// `getExprInfoAt(idx)` (`Analyses/PropagationAnalysis.h:254`) — the bucket unit `idx` reads.
    ///
    /// ⛔ `None` COVERS BOTH OF THE REFERENCE'S ABORTS: `.at()` out of range and the null bucket its
    /// callers guard with `DT_CHECK_MSG(expr_info, "Expecting valid ExprInfo for unit")`
    /// (`ScalarSimplifications.cpp:344`).
    #[must_use]
    pub fn expr_info_at(&self, unit: UnitIndex) -> Option<&PropagatedExpr> {
        let bucket = *self.buckets.get(unit.0 as usize)?;
        self.exprs.get(bucket)?.as_ref()
    }
}

/// THE `RegisterPressure` A PASS CONSTRUCTS OVER ONE UNIT — a trait, for the same reason
/// [`ExpressionEvaluator`] is one: the analysis is not in this campaign and a test must still be able
/// to state its answers.
///
/// ⛔ `Analyses/RegisterPressureAnalysis.{h,cpp}` IS OUT OF CAMPAIGN SCOPE — and so is the
/// `Liveness` every constructor of it takes (`RegisterPressureAnalysis.h:54-56`) — so the crate's only
/// implementation is [`OutOfScopeRegisterPressure`] and its one method is a `todo!`.
///
/// ⭐ ONE METRIC, NOT THE `enum class Metric`: `kNumRegisters` is `kDefault` (`:37-39`) and is the only
/// one any ported caller asks for; the other three need the `dcc_ctx` this seam does not carry (`:53`).
pub trait RegisterPressure {
    /// `getOrComputeRegisterPressure(locale, Metric::kNumRegisters)` (`:71`) — how many registers of
    /// `locale` the unit is estimated to need.
    fn num_registers(&mut self, locale: RegType) -> u32;
}

/// THE ONE CRATE IMPLEMENTATION: register pressure is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeRegisterPressure;

impl RegisterPressure for OutOfScopeRegisterPressure {
    fn num_registers(&mut self, _locale: RegType) -> u32 {
        todo!(
            "RegisterPressure::getOrComputeRegisterPressure (Analyses/RegisterPressureAnalysis.h:71) — out of campaign scope"
        )
    }
}
