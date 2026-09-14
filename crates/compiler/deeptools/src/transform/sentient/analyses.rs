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

use std::collections::BTreeMap;

use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient::{RegIndex, RegType};
use crate::islands::sentient::dialects::{Op, Val};
use crate::transform::sentient::ForRef;
use crate::transform::sentient::canonicalize_xrf_pointers::XrfMinExpr;
use crate::transform::sentient::local_region_splitting_for_value_commoning::MaxRegNum;
use crate::transform::sentient::port_assignment::{DataId, PortId};
use crate::units::DfirUnit;

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

impl EvaluatedValue {
    /// `raw_ostream &operator<<(raw_ostream &, const EvaluatedValue &)`
    /// (`Analyses/ExpressionEvaluatorUtils.h:405`) — the analysis's own rendering of the arena entry
    /// this handle names, which is what every `dump()` in `AddressPinningAndToggle.cpp` prints.
    ///
    /// ⛔ NOT DERIVABLE FROM THE HANDLE, which is why it is not a `Display`: the digits live in the
    /// evaluator's memoisation arena, and printing `self.0` would put a slot number where the
    /// reference prints an address.
    #[must_use]
    pub fn rendered(self) -> String {
        todo!(
            "EvaluatedValue::operator<< (Analyses/ExpressionEvaluatorUtils.h:405) — out of campaign scope"
        )
    }
}

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

    /// `ExpressionEvaluator::getConstant` (`Analyses/ExpressionEvaluatorUtils.h:335`) for the DECODED
    /// flavour — the `getConstant(0)` a merging block starts its running increment from, which e528
    /// must also be able to measure against the LRF range.
    fn constant_evaluation(&mut self, value: i64) -> Evaluation {
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

    /// `ExpressionEvaluator::evaluateDivideByConst`
    /// (`Analyses/ExpressionEvaluatorUtils.h:294`) — `ev / by`.
    fn evaluate_divide_by_const(&mut self, ev: EvaluatedValue, by: i64) -> EvaluatedValue {
        let _ = (ev, by);
        todo!(
            "ExpressionEvaluator::evaluateDivideByConst (Analyses/ExpressionEvaluatorUtils.h:294) — out of campaign scope"
        )
    }

    /// `ExpressionEvaluator::evaluateMultiplyByConst` (`Analyses/ExpressionEvaluatorUtils.h:285`) for
    /// the DECODED flavour — how e530 turns a derived `B - c` into `B + (c * -1)` before measuring it
    /// against the LRF range.
    ///
    /// ⛔ NOT `_handle`-SUFFIXED THE OTHER WAY ROUND: the handle flavour took the reference's own name
    /// first, so this one is spelled short rather than renaming its sixteen call sites.
    fn multiply_by_const(&mut self, ev: &Evaluation, by: i64) -> Evaluation {
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

    /// `EvaluatedValue`'S OWN OFFSETS, READ FROM A STORED HANDLE
    /// (`Analyses/ExpressionEvaluatorUtils.h:63-126`) — the read twin of
    /// [`ExpressionEvaluator::build_offset_value_of`], for the range test
    /// `ScalarOpMerging::markFieldUnrollingCandidates` (`:1249`) applies to a merging increment the
    /// block RECORDED rather than one it has just evaluated.
    ///
    /// ⛔ THE DECODE IS THE ANALYSIS'S, NOT THE PORTER'S: the arena entry the handle names belongs to
    /// `Analyses/ExpressionEvaluatorUtils`, out of campaign scope, so this states the seam and does
    /// not invent it — exactly as [`ExpressionEvaluator::is_any_val_less_than`] already reads a
    /// handle's values through the trait rather than reconstructing them.
    fn evaluation_of(&mut self, handle: EvaluatedValue) -> Evaluation {
        let _ = handle;
        todo!(
            "EvaluatedValue offsets (Analyses/ExpressionEvaluatorUtils.h:63) — out of campaign scope"
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

/// ONE DATA TRANSFER'S BYTE-ADDRESSABLE ADDRESSES, AS THE DYNAMIC PINNING SCHEME IS HANDED THEM —
/// `DynamicPinningSchemeManager::EVAddressInfo` (`Analyses/AddressPinningScheme.h:279-295`), built
/// per base address by `computeAddressInfoList` (`AddressPinningAndToggle.cpp:1729-1731`).
///
/// ⛔ DATA, NOT AN ANALYSIS, exactly as [`CorrelatedEquivClass`] is: `Analyses/AddressPinningScheme`
/// is out of campaign scope, so nothing here COMPUTES a scheme — this is the record a pass
/// accumulates and [`PinningSchemeManager`] consumes.
///
/// ⭐ THE MINIMUM MUTABLE ADDRESS IS NOT A FIELD: `computeAddressInfoList` evaluates
/// `ba_min_mut_addr_ev` and prints it, then passes only the immutable and MAX addresses (`:1730`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvAddressInfo {
    /// `ba_immut_addr_ev_` — the transfer's immutable address, in bytes.
    pub ba_immut_addr_ev: EvaluatedValue,
    /// `ba_max_mut_addr_ev_` — the HIGHEST mutable address it reaches, in bytes.
    pub ba_max_mut_addr_ev: EvaluatedValue,
    /// `is_toggle_` — `immut_dtd->isToggle()`, which is what makes the pair a toggling one.
    pub is_toggle: bool,
    /// `region_op_` and `region_num_` together — see [`RegionSite`].
    pub region: RegionSite,
}

/// THE `const PinningSchemeManager&` A PASS IS HANDED — a trait, for the reason
/// [`ExpressionEvaluator`] is one: `Analyses/AddressPinningScheme.{h,cpp}` is not in this campaign and
/// a test must still be able to state which pinned address it chose.
pub trait PinningSchemeManager {
    /// `PinningSchemeManager::getMaxNumRegisters()` (`Analyses/AddressPinningScheme.h:194-196`) — the
    /// IMMUTABLE-address registers the manager's unit has: the LBR file's `maxNum` for LX (0 on
    /// SEN1P5) and the EBR file's otherwise (`AddressPinningScheme.cpp:57-67`).
    ///
    /// ⛔ A `todo!` THOUGH EVERY NUMBER IN IT IS `sysDef`'s: `UnitAndHardwareInfo` is filled by the
    /// out-of-scope `setUnitHardwareInfo`, and its getters all open `DT_CHECK(evaluated_)`.
    fn max_num_registers(&self) -> MaxRegNum {
        todo!(
            "PinningSchemeManager::getMaxNumRegisters (Analyses/AddressPinningScheme.h:194) — out of campaign scope"
        )
    }

    /// `PinningSchemeManager::getMemoryUnit()` (`Analyses/AddressPinningScheme.h:198-200`) — LX or
    /// HBM, the memory this manager's schemes pin addresses in, as [`Self::eval`] was handed it.
    fn memory_unit(&self) -> DfirUnit {
        todo!(
            "PinningSchemeManager::getMemoryUnit (Analyses/AddressPinningScheme.h:198) — out of campaign scope"
        )
    }

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

    /// `PinningSchemeManager::overflowsRegister(addr_ev, element_size_in_bits)`
    /// (`Analyses/AddressPinningScheme.h:236-238`) — whether LAR/EAR is too narrow for the signed
    /// element addresses in `addr_ev`: *"Returns false if they all fit and true otherwise."*
    ///
    /// ⛔ `true` IS THE ABORT, NOT THE SUCCESS. All four call sites are
    /// `DT_CHECK(!ps_manager_.overflowsRegister(..))` (`AddressPinningAndToggle.cpp:1927`, `:2030`,
    /// `:2101`, `:2156`), so a port that reads the name as "fits" inverts every one of them.
    fn overflows_register(&self, addr_ev: EvaluatedValue, element_size: Bits) -> bool {
        let _ = (addr_ev, element_size);
        todo!(
            "PinningSchemeManager::overflowsRegister (Analyses/AddressPinningScheme.h:236) — out of campaign scope"
        )
    }

    /// `PinningSchemeManager::findMatchingPinnedAddr(ev, element_size_in_bits)`
    /// (`Analyses/AddressPinningScheme.h:245-246`) — for each unit of the manager's scheme, WHERE in
    /// that unit's pinning scheme the address `ev` was pinned.
    ///
    /// ⛔ `None` IS THE REFERENCE'S `-1`, and `using IndexTy = int64_t;  // must be signed` (`:23`) is
    /// signed for exactly that: *"for any unit, if no matching pinned address is found, map that unit
    /// to -1"* (`:242-243`). A non-negative entry is a register index, which is what its one caller
    /// writes it as.
    /// ⭐ ON THE BASE MANAGER, not on `StaticPinningSchemeManager`: the derived class its caller names
    /// declares no override (`:258-270`), so one seam serves both.
    fn find_matching_pinned_addr(
        &self,
        ev: EvaluatedValue,
        element_size: Bits,
    ) -> Vec<(Val, Option<RegIndex>)> {
        let _ = (ev, element_size);
        todo!(
            "PinningSchemeManager::findMatchingPinnedAddr (Analyses/AddressPinningScheme.h:245) — out of campaign scope"
        )
    }

    /// `StaticPinningSchemeManager::eval(dcc_ext_context, evaluator, memory_unit)`
    /// (`Analyses/AddressPinningScheme.h:267-268`) — computes the per-unit static pinned address
    /// schemes, which is what makes the two `find*PinnedAddr` seams answerable.
    ///
    /// ⛔ `&mut self` AND ON THIS TRAIT THOUGH IT IS THE **DERIVED** CLASS'S: it is non-const, it is
    /// the only mutator any ported pass calls on a manager, and its callers hold the static manager —
    /// so one seam serves both here for the reason [`Self::find_matching_pinned_addr`] gives.
    /// ⭐ `dccExtContext()` IS DROPPED, NOT FORGOTTEN: it is the arch/hardware description, which this
    /// crate carries as the `A: Arch` parameter its caller is generic over — and a generic method here
    /// would make the trait no longer `dyn`-compatible.
    fn eval(&mut self, evaluator: &mut dyn ExpressionEvaluator, memory_unit: DfirUnit) {
        let _ = (evaluator, memory_unit);
        todo!(
            "StaticPinningSchemeManager::eval (Analyses/AddressPinningScheme.h:267) — out of campaign scope"
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

    /// `haveIbuffSpace(ctx, dataflow::ProgramUnitOp unit)` (`Analyses/InstructionEstimation.h:64`) —
    /// whether the unit's instructions still fit its instruction buffer.
    fn have_ibuff_space(&mut self, unit: &[Op]) -> bool;

    /// `getRemainingIbuffSpace(ctx, unit)` (`Analyses/InstructionEstimation.h:69`) — what is LEFT of
    /// the instruction buffer, which is why [`InstructionCount`] is signed.
    ///
    /// ⭐ DEFAULTED SO A TEST DOUBLE NEED NOT REPEAT THE REFUSAL: one unit asks
    /// (`e514_findOptimalNWaySplits`), and only to log the comparison.
    fn remaining_ibuff_space(&mut self, unit: &[Op]) -> InstructionCount {
        let _ = unit;
        todo!(
            "InstructionEstimatorImpl::getRemainingIbuffSpace (Analyses/InstructionEstimation.h:69) — out of campaign scope"
        )
    }
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

    fn have_ibuff_space(&mut self, _unit: &[Op]) -> bool {
        todo!(
            "InstructionEstimatorImpl::haveIbuffSpace (Analyses/InstructionEstimation.h:64) — out of campaign scope"
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
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeCandidateEvaluator`]. The three methods the `Driver`'s
/// phases call are declared and nothing else is.
pub trait CandidateEvaluator {
    /// `evaluateLocally(candidates)` — weights, sorts and re-prioritises IN PLACE.
    fn evaluate_locally(&mut self, candidates: &mut Vec<Candidate>);

    /// `evaluateGlobally(candidates)` (`Evaluator.h:56`) — the same, once the grouper has replaced
    /// grouped locals with global candidates.
    fn evaluate_globally(&mut self, candidates: &mut Vec<Candidate>);

    /// `mergeInto(result, sublist)` (`Evaluator.h:66`) — inserts `sublist` keeping `result` sorted.
    fn merge_into(&mut self, result: &mut Vec<Candidate>, sublist: &[Candidate]);
}

/// THE `SelectorInterface &` THE `Driver` IS HANDED (`RegisterInitialization/Selector.h:38`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeCandidateSelector`].
pub trait CandidateSelector {
    /// `selectLocally(local, global, core)` (`Selector.h:68`) — PURGES both lists in place; neither
    /// can grow.
    fn select_locally(
        &mut self,
        local: &mut Vec<Candidate>,
        global: &mut Vec<Candidate>,
        core: Val,
    );

    /// `selectGlobally(candidates)` (`Selector.h:81`) — the global phase's purge of the one merged
    /// list.
    fn select_globally(&mut self, candidates: &mut Vec<Candidate>);
}

/// THE `UniformGrouper &` THE `Driver` IS HANDED (`RegisterInitialization/UniformGrouper.h:23`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeUniformGrouper`]. ⭐ A TRAIT THOUGH THE REFERENCE'S CLASS
/// IS CONCRETE, for the reason [`ExpressionEvaluator`] is one: a test must be able to state what the
/// grouping did.
pub trait UniformGrouper {
    /// `run(candidates)` (`UniformGrouper.h:36`) — IN PLACE, and the one phase that may GROW the list:
    /// grouped locals come out and global candidates go in.
    fn run(&mut self, candidates: &mut Vec<Candidate>);
}

/// THE `Transformer &` THE `Driver` IS HANDED (`RegisterInitialization/Transformer.h:25`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeTransformer`]. ⭐ A TRAIT for the reason
/// [`UniformGrouper`] is one.
pub trait Transformer {
    /// `run(candidates)` (`Transformer.h:34`) — the phase that rewrites the IR; the list is read-only.
    fn run(&mut self, candidates: &[Candidate]);
}

/// THE `const UniformGroupAnalyzer &` THE `Driver` IS HANDED
/// (`Analyses/UniformGroupAnalysis.h:57`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeUniformGroups`]. Only `getGroupLeaders` is declared.
pub trait UniformGroups {
    /// `getGroupLeaders()` (`Analyses/UniformGroupAnalysis.h:67`) — one unit value per exclusive
    /// group.
    fn group_leaders(&self) -> Vec<Val>;

    /// `isGroupLeader(unit)` (`Analyses/UniformGroupAnalysis.h:68`) — is `unit` a leader, which the
    /// analysis answers as "not a key of `non_leader_group_members_`".
    fn is_group_leader(&self, unit: Val) -> bool;

    /// `getGroupMembersLedBy(leader)` (`Analyses/UniformGroupAnalysis.h:72`) — the non-leader units
    /// `leader` speaks for, `leader` itself excluded.
    fn group_members_led_by(&self, leader: Val) -> Vec<Val>;

    /// `collectExclusiveGroupLeaders()` (`Analyses/UniformGroupAnalysis.h:66`) — RUNS the analysis,
    /// which is what makes the three getters above answerable at all.
    ///
    /// ⛔ `&mut self` AND DEFAULTED, unlike its three read-only siblings: it is the analysis's one
    /// mutator, every caller constructs the analyzer itself, and a defaulted seam keeps the fakes that
    /// only need the getters implementable.
    fn collect_exclusive_group_leaders(&mut self) {
        todo!(
            "UniformGroupAnalyzer::collectExclusiveGroupLeaders (Analyses/UniformGroupAnalysis.h:66) — out of campaign scope"
        )
    }
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

    fn evaluate_globally(&mut self, _candidates: &mut Vec<Candidate>) {
        todo!(
            "EvaluatorInterface::evaluateGlobally (RegisterInitialization/Evaluator.h:56) — out of campaign scope"
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

    fn select_globally(&mut self, _candidates: &mut Vec<Candidate>) {
        todo!(
            "SelectorInterface::selectGlobally (RegisterInitialization/Selector.h:81) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`UniformGrouper`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeUniformGrouper;

impl UniformGrouper for OutOfScopeUniformGrouper {
    fn run(&mut self, _candidates: &mut Vec<Candidate>) {
        todo!(
            "UniformGrouper::run (RegisterInitialization/UniformGrouper.h:36) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`Transformer`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeTransformer;

impl Transformer for OutOfScopeTransformer {
    fn run(&mut self, _candidates: &[Candidate]) {
        todo!("Transformer::run (RegisterInitialization/Transformer.h:34) — out of campaign scope")
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

/// HOW MANY COLOURS A COLOURING MAY USE — `doGraphColoring(int)`'s only argument
/// (`Analyses/GraphColoring.hpp:82`).
///
/// ⭐ A NEWTYPE BECAUSE THE TWO CALLERS COUNT DIFFERENT THINGS: for port assignment it is the three
/// compute ports ([`PortId::COUNT`]), for register allocation the locale's register count
/// ([`MaxRegNum`]) — and a bare `3` beside a bare register count is the confusion this crate's
/// newtypes exist to make an E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NumColors(pub i32);

/// `doGraphColoring(int, bool use_greedy_allocator = false)`'S SECOND ARGUMENT
/// (`Analyses/GraphColoring.hpp:82`) — a closed set, so an `enum` and not the reference's `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreedyAllocator {
    /// The default the port assignment relies on (`PortAssignment.cpp:736`).
    No,
    /// `use_greedy_allocator = true`.
    Yes,
}

/// THE `GraphColoring` A PASS OWNS — a trait, for the same reason [`ExpressionEvaluator`] is one: the
/// analysis is not in this campaign and a test must still be able to observe what a pass asks it.
///
/// ⛔ `Analyses/GraphColoring.{hpp,cpp}` IS OUT OF CAMPAIGN SCOPE. Only the methods a ported unit
/// actually calls are declared; `getOrAddNode`, `addBidirectionalEdge`, `addSameColorEdge` and
/// `doGraphColoring` belong to `e339`, `e608` and `e382` and are added by those units.
// ⭐ `pub(crate)`, UNLIKE THE OTHER SEAMS IN THIS FILE: the colour and the node index are
// [`PortId`] and [`DataId`], which port assignment owns and keeps crate-private, and this trait's
// only consumer is that pass's own `pub(crate) struct PortAssignment<G: ColoringGraph>`.
pub(crate) trait ColoringGraph {
    /// `GraphColoring::clear()` (`Analyses/GraphColoring.hpp:52-60`).
    ///
    /// ⛔⛔ NOT A RE-DEFAULT-CONSTRUCTION, AND THE DIFFERENCE OUTLIVES A PROGRAM UNIT. It deletes and
    /// clears `nodes_`, then clears `node_ids_` and `edges_` — and leaves `same_color_edges_` and
    /// `max_node_id_` STANDING, so both survive into the next unit the pass visits even though the
    /// constructor's own `clear()` ran on an empty object. Whoever gives this an interior must never
    /// write `*self = Self::default()`.
    fn clear(&mut self);

    /// `GraphNode *getOrAddNode(int index)` (`Analyses/GraphColoring.hpp:64`) — the node for `index`,
    /// created if the graph has none yet.
    ///
    /// ⭐ THE RETURNED `GraphNode *` IS DROPPED: every ported caller uses this for its side effect, so
    /// the node's own interior stays inside the out-of-scope analysis.
    fn get_or_add_node(&mut self, index: DataId);

    /// `addBidirectionalEdge(int node1, int node2)` (`Analyses/GraphColoring.hpp:80`) — an interference
    /// edge recorded on BOTH nodes.
    fn add_bidirectional_edge(&mut self, node1: DataId, node2: DataId);

    /// `getOrAddNode(index)->addValidValues(values)` (`Analyses/GraphColoring.hpp:34`) — the colours
    /// this node may take.
    ///
    /// ⛔⛔ IT **REPLACES**, IT DOES NOT ADD. `addValidValues`' `addon` parameter defaults to `false`
    /// and every port-assignment caller leaves it there, so the third argument's whole effect is
    /// `setValidValues` (`Analyses/GraphColoring.cpp:44-57`) — which is why `e339`'s later
    /// `addValidValues({2})` on a node it already constrained is a narrowing and not a union.
    /// ⭐ THE LIST IS KEPT SORTED AND UNIQUE by the analysis itself, and [`PortId`] is `Ord`.
    fn set_valid_values(&mut self, index: DataId, values: &[PortId]);

    /// `getOrAddNode(index)->getValidValues()` (`Analyses/GraphColoring.hpp:36`).
    ///
    /// ⭐ `&mut self` BECAUSE `getOrAddNode` MAY CREATE THE NODE: asking an unconstrained node what it
    /// may take adds it to the graph, which `e340` does on every `DataTransferOnly` mac.
    fn valid_values(&mut self, index: DataId) -> Vec<PortId>;

    /// `int getNumNodes() const` (`Analyses/GraphColoring.hpp:118`) — `nodes_.size()`.
    ///
    /// ⛔ IT IS THE NODE MAP AND NOT `max_node_id_`, which `clear()` leaves standing: a graph cleared
    /// between units answers 0 here even though it still remembers the highest id it ever issued.
    fn num_nodes(&self) -> usize;

    /// `std::map<int, int> doGraphColoring(int, bool)` (`Analyses/GraphColoring.hpp:82`) — the colour
    /// each node was given.
    ///
    /// ⛔ THE COLOUR IS A [`PortId`] BECAUSE [`DataId`] IS ALREADY THE PORT ASSIGNMENT'S NODE INDEX:
    /// this seam is the graph AS PORT ASSIGNMENT HOLDS IT, and the register allocator's own colouring
    /// reaches the same analysis through [`RegisterGraphs`].
    /// ⛔ A NODE THE COLOURING COULD NOT PLACE IS SIMPLY ABSENT FROM THE MAP, which is what
    /// `getPortAttr`'s `port_assignment_.at(opID)` throws on (`:207`).
    fn do_graph_coloring(
        &mut self,
        num_colors: NumColors,
        greedy: GreedyAllocator,
    ) -> BTreeMap<DataId, PortId>;
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeColoringGraph;

impl ColoringGraph for OutOfScopeColoringGraph {
    fn clear(&mut self) {
        todo!("GraphColoring::clear (Analyses/GraphColoring.hpp:52) — out of campaign scope")
    }

    fn get_or_add_node(&mut self, _index: DataId) {
        todo!("GraphColoring::getOrAddNode (Analyses/GraphColoring.hpp:64) — out of campaign scope")
    }

    fn add_bidirectional_edge(&mut self, _node1: DataId, _node2: DataId) {
        todo!(
            "GraphColoring::addBidirectionalEdge (Analyses/GraphColoring.hpp:80) — out of campaign scope"
        )
    }

    fn set_valid_values(&mut self, _index: DataId, _values: &[PortId]) {
        todo!(
            "GraphNode::addValidValues (Analyses/GraphColoring.hpp:34) — out of campaign scope"
        )
    }

    fn valid_values(&mut self, _index: DataId) -> Vec<PortId> {
        todo!(
            "GraphNode::getValidValues (Analyses/GraphColoring.hpp:36) — out of campaign scope"
        )
    }

    fn num_nodes(&self) -> usize {
        todo!("GraphColoring::getNumNodes (Analyses/GraphColoring.hpp:118) — out of campaign scope")
    }

    fn do_graph_coloring(
        &mut self,
        _num_colors: NumColors,
        _greedy: GreedyAllocator,
    ) -> BTreeMap<DataId, PortId> {
        todo!(
            "GraphColoring::doGraphColoring (Analyses/GraphColoring.hpp:82) — out of campaign scope"
        )
    }
}

/// A NODE OF ONE LOCALE'S REGISTER INTERFERENCE GRAPH — the `int` a `RegisterGraphs` colouring is
/// keyed by, which is [`Liveness::operand_to_index`]'s index for the value.
///
/// ⭐ NOT [`DataId`]: that is port assignment's node index over the same out-of-scope graph class, and
/// a register graph node beside a compute-port graph node is exactly the confusion a newtype makes an
/// E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegNode(pub i32);

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

    /// `buildGraphs(dcc_ext_ctx, liverange, unit, target_locale, coreunit, dumpLiveRange)`
    /// (`Analyses/GraphColoring.hpp:154-157`) — one interference graph per locale over `unit`'s live
    /// ranges, or over `target_locale` alone.
    ///
    /// ⭐ `Liveness &` IS NON-CONST THERE, hence `&mut`. `coreunit` is the reference's nullable
    /// `mlir::Value`; the `dumpLiveRange` debug flag is dropped, being output and not behaviour.
    fn build_graphs<L: Liveness>(
        &mut self,
        liveness: &mut L,
        unit: &[Op],
        locale: RegType,
        coreunit: Option<Val>,
    );

    /// `createSameColorEdgeEqClasses(liveness, target_locale)`
    /// (`Analyses/GraphColoring.hpp:168-170`) — folds the virtual assignments `liveness` holds into
    /// `ec_map_`'s equivalence classes.
    fn create_same_color_edge_eq_classes<L: Liveness>(&mut self, liveness: &mut L, locale: RegType);

    /// `buildHyperGraph(targetLocale, buildHyperGraph)` (`Analyses/GraphColoring.hpp:158-159`) —
    /// collapses each equivalence class to one hyper-node. The second parameter is the reference's own
    /// `bool buildHyperGraph` default, dropped: no ported call site passes it.
    fn build_hyper_graph(&mut self, locale: RegType);

    /// `fastCheckColorability(num_colors, locale)` (`Analyses/GraphColoring.hpp:167`) — whether
    /// `locale`'s graph still colours in `num_colors` registers. ⭐ NON-CONST there, hence `&mut`.
    fn fast_check_colorability(&mut self, num_colors: MaxRegNum, locale: RegType) -> bool;

    /// `doGraphColorOnLocale(dcc_ext_ctx, locale, unit, use_greedy_allocator)`
    /// (`Analyses/GraphColoring.hpp:163`) — which register index each node of `locale`'s hyper-graph
    /// was given, over the register count `locale` has on this machine.
    ///
    /// ⛔ TRAP: THE RESULT IS A `std::map` ITS CALLER SUBSCRIPTS, so a node the colouring never placed
    /// reads back as register 0 rather than absent — see `e382`'s `Colorings::index`.
    fn do_graph_color_on_locale(
        &mut self,
        locale: RegType,
        unit: &[Op],
        greedy: GreedyAllocator,
    ) -> BTreeMap<RegNode, RegIndex>;
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeRegisterGraphs;

impl RegisterGraphs for OutOfScopeRegisterGraphs {
    fn clean(&mut self) {
        todo!("RegisterGraphs::clean (Analyses/GraphColoring.hpp:179) — out of campaign scope")
    }

    fn build_graphs<L: Liveness>(
        &mut self,
        _liveness: &mut L,
        _unit: &[Op],
        _locale: RegType,
        _coreunit: Option<Val>,
    ) {
        todo!(
            "RegisterGraphs::buildGraphs (Analyses/GraphColoring.hpp:154) — out of campaign scope"
        )
    }

    fn create_same_color_edge_eq_classes<L: Liveness>(
        &mut self,
        _liveness: &mut L,
        _locale: RegType,
    ) {
        todo!(
            "RegisterGraphs::createSameColorEdgeEqClasses (Analyses/GraphColoring.hpp:168) — out of campaign scope"
        )
    }

    fn build_hyper_graph(&mut self, _locale: RegType) {
        todo!(
            "RegisterGraphs::buildHyperGraph (Analyses/GraphColoring.hpp:158) — out of campaign scope"
        )
    }

    fn fast_check_colorability(&mut self, _num_colors: MaxRegNum, _locale: RegType) -> bool {
        todo!(
            "RegisterGraphs::fastCheckColorability (Analyses/GraphColoring.hpp:167) — out of campaign scope"
        )
    }

    fn do_graph_color_on_locale(
        &mut self,
        _locale: RegType,
        _unit: &[Op],
        _greedy: GreedyAllocator,
    ) -> BTreeMap<RegNode, RegIndex> {
        todo!(
            "RegisterGraphs::doGraphColorOnLocale (Analyses/GraphColoring.hpp:163) — out of campaign scope"
        )
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

/// ONE CORRELATED EQUIVALENCE CLASS OF LOOP ITERATOR ARGUMENTS — `CorrelatedEquivClass`
/// (`Analyses/CorrelationAnalysis.h:140`) reduced to the two things a ported pass reads of it.
///
/// ⛔ DATA, NOT AN ANALYSIS: `Analyses/CorrelationAnalysis.{h,cpp}` is out of campaign scope, so
/// nothing here COMPUTES a class — this is the identity of an answer the pass is handed, which is what
/// lets e467's replacement be observed without inventing the correlation walk that finds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrelatedEquivClass {
    /// `getHeadInfo()->getIterArg()` — the iterator argument every member is an offset FROM.
    pub head_iter_arg: Val,
    /// `getIterArgToCorrelation()` — each other member paired with its constant offset from the head.
    ///
    /// ⛔ THE HEAD IS NOT IN HERE, which is why e467 never replaces an argument by itself.
    pub correlations: Vec<(Val, i64)>,
}

/// `CorrelatedEquivClassContainer` (`Analyses/CorrelationAnalysis.h:290`) — the classes one loop's
/// iterator arguments fall into.
///
/// ⭐ `isEmpty()` IS `classes.is_empty()`, and the container POINTER the reference also guards
/// against (`ReuseLoopIteratorArguments.cpp:267`) is the caller's `Option`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CorrelatedEquivClasses {
    /// `getCorrelatedEquivClasses()`, in the order the container hands them out.
    pub classes: Vec<CorrelatedEquivClass>,
}

impl CorrelatedEquivClasses {
    /// `correlated_equiv_classes->isEmpty()`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }
}

/// WHETHER A CORRELATION ANALYSIS ANSWERED — `isCorrelationSuccessful()`'s `LogicalResult`
/// (`Analyses/CorrelationAnalysis.h:330`), which reports THE ANALYSIS'S state and not a failure of the
/// pass reading it, so the crate's refusal ban is not in play.
///
/// ⛔ THE REFERENCE DOES NOT `return` ON `Failed`: `emitError` + `signalPassFailure()` mark the
/// pipeline and fall straight through into the round that needed the analysis
/// (`ReuseLoopIteratorArguments.cpp:151-155`). A port that returned here would run less than the
/// reference does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Correlation {
    /// `LogicalResult::success()`.
    Succeeded,
    /// `LogicalResult::failure()`.
    Failed,
}

/// A `CorrelationAnalysisBase` A PASS IS HANDED — a trait for the same reason [`ExpressionEvaluator`]
/// is one: the analysis is not in this campaign, and a test must still be able to state its answers.
///
/// ⛔ `Analyses/CorrelationAnalysis.{h,cpp}` IS OUT OF CAMPAIGN SCOPE. `ToggleCorrelationAnalysis`
/// and `AffineExpressionCorrelationAnalysis` are two DERIVED classes reached through this same base
/// (`:150`, `:246`), which is why one trait carries both and the crate has one `OutOfScope*` per
/// derived class rather than per method.
/// ⭐ EACH IS CONSTRUCTED OVER ONE `dataflow.program_unit`, so a value handed in is already scoped to
/// the unit its reader is walking; the affine one also takes the propagation analysis (`:248`).
pub trait CorrelationAnalysis {
    /// `isCorrelationSuccessful()` — whether the constructor's walk completed.
    fn is_correlation_successful(&mut self) -> Correlation;

    /// `getCorrelatedEquivClasses(sentient::ForOp)` — the classes recorded for one loop, keyed by the
    /// loop itself.
    ///
    /// ⭐ `None` IS THE REFERENCE'S NULL CONTAINER POINTER, which its readers guard against
    /// (`ReuseLoopIteratorArguments.cpp:267`).
    fn correlated_equiv_classes(&self, for_op: ForRef) -> Option<&CorrelatedEquivClasses>;
}

/// THE CRATE IMPLEMENTATION OF `ToggleCorrelationAnalysis`: not ported, so asking it anything is a
/// `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeToggleCorrelation;

impl CorrelationAnalysis for OutOfScopeToggleCorrelation {
    fn is_correlation_successful(&mut self) -> Correlation {
        todo!(
            "ToggleCorrelationAnalysis::isCorrelationSuccessful (Analyses/CorrelationAnalysis.h:150) — out of campaign scope"
        )
    }

    fn correlated_equiv_classes(&self, _for_op: ForRef) -> Option<&CorrelatedEquivClasses> {
        todo!(
            "ToggleCorrelationAnalysis::getCorrelatedEquivClasses (Analyses/CorrelationAnalysis.h:150) — out of campaign scope"
        )
    }
}

/// THE CRATE IMPLEMENTATION OF `AffineExpressionCorrelationAnalysis`: not ported, so asking it
/// anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeAffineExpressionCorrelation;

impl CorrelationAnalysis for OutOfScopeAffineExpressionCorrelation {
    fn is_correlation_successful(&mut self) -> Correlation {
        todo!(
            "AffineExpressionCorrelationAnalysis::isCorrelationSuccessful (Analyses/CorrelationAnalysis.h:246) — out of campaign scope"
        )
    }

    fn correlated_equiv_classes(&self, _for_op: ForRef) -> Option<&CorrelatedEquivClasses> {
        todo!(
            "AffineExpressionCorrelationAnalysis::getCorrelatedEquivClasses (Analyses/CorrelationAnalysis.h:246) — out of campaign scope"
        )
    }
}

/// AN `ExprInfoMap` THE PROPAGATION ANALYSIS OWNS — an identity, not the map, for exactly the reason
/// [`EvaluatedValue`] is one: `getAffineExpression` returns a borrowed
/// `PropagationAnalysisImpl::ExprInfoMap *` into the analysis's own memoisation
/// (`Analyses/PropagationAnalysis.h:308`), and the `AffineMap` inside each `ExprInfo` is MLIR affine
/// machinery this crate does not carry at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprInfoMap(pub u32);

/// THE `Liveness&` A PASS IS HANDED — a trait for the same reason [`ExpressionEvaluator`] is one:
/// the analysis is not in this campaign, and a test must still be able to observe WHICH values a
/// ported pass promotes.
///
/// ⛔ `Analyses/Liveness.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only implementation is
/// [`OutOfScopeLiveness`] and every method of it is a `todo!`.
pub trait Liveness {
    /// `updateLiveRangesForProgramHeaderPromotion(candidate)` (`Analyses/Liveness.h:130`) — widens
    /// `candidate`'s live range to the whole program because it is about to live in the header.
    ///
    /// ⭐ THE PLURAL OVERLOAD (`:131-135`) IS AN INLINE `for` OVER THIS ONE, so it is a loop at the
    /// call site and not a second method.
    fn update_live_ranges_for_program_header_promotion(&mut self, candidate: Val);

    /// `isLiveRangeOverlaps(val1, val2)` (`Analyses/Liveness.h:111`) — do the two values' live
    /// ranges intersect anywhere, i.e. may they NOT share one register.
    fn is_live_range_overlaps(&self, val1: Val, val2: Val) -> bool;

    /// `clear(clear_virtual_assigns)` (`Analyses/Liveness.h:136`, `Analyses/Liveness.cpp:991-1002`) —
    /// empties the seven index and live-range maps, and `va_` only when asked.
    fn clear(&mut self, virtual_assigns: VirtualAssigns);

    /// `computeRegisterLiveRange(unit)` (`Analyses/Liveness.h:88`) — indexes every op in the program
    /// unit and rebuilds the live ranges over those indices.
    fn compute_register_live_range(&mut self, unit: &[Op]);

    /// `addVirtualAssignOptional(set_of_subsets)` (`Analyses/Liveness.h:118-119`) — links every pair
    /// within each subset as MAY share a register.
    fn add_virtual_assign_optional(&mut self, set_of_subsets: &[Vec<Val>]);

    /// `addVirtualAssignEnforced(set_of_subsets)` (`Analyses/Liveness.h:124-125`) — links each pair as
    /// MUST share a register. ⭐ THE OVERLOAD THIS IS, of the three, takes PAIRS and not subsets.
    fn add_virtual_assign_enforced(&mut self, set_of_pairs: &[(Val, Val)]);

    /// `getOperandToIndex()[value]` (`Analyses/Liveness.h:143`) — which interference-graph node
    /// `value` is, i.e. the key a [`RegisterGraphs`] colouring answers for it.
    ///
    /// ⛔ TRAP: THE REFERENCE SUBSCRIBES A `DenseMap`, so a value liveness never indexed is INSERTED
    /// with node 0 rather than refused — the seam keeps that totality and returns a [`RegNode`] always.
    fn operand_to_index(&mut self, value: Val) -> RegNode;
}

/// WHETHER A `Liveness::clear` ALSO DROPS THE VIRTUAL ASSIGNMENTS — `clear`'s defaulted `bool`
/// (`Analyses/Liveness.h:136`), which is two different erasures and not a flag.
///
/// ⭐ BOTH CASES OCCUR: `Transformer.cpp:110` passes `true`, and `OldRegisterInitialization.cpp:968`
/// and `:1050` take the default — so `e452`'s clear KEEPS the assignments `e451` established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualAssigns {
    /// `clear_virtual_assigns == false`: `va_` survives.
    Kept,
    /// `clear_virtual_assigns == true`: `va_.clear()` too.
    Cleared,
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

    fn clear(&mut self, _virtual_assigns: VirtualAssigns) {
        todo!("Liveness::clear (Analyses/Liveness.h:136) — out of campaign scope")
    }

    fn compute_register_live_range(&mut self, _unit: &[Op]) {
        todo!("Liveness::computeRegisterLiveRange (Analyses/Liveness.h:88) — out of campaign scope")
    }

    fn add_virtual_assign_optional(&mut self, _set_of_subsets: &[Vec<Val>]) {
        todo!(
            "Liveness::addVirtualAssignOptional (Analyses/Liveness.h:118) — out of campaign scope"
        )
    }

    fn add_virtual_assign_enforced(&mut self, _set_of_pairs: &[(Val, Val)]) {
        todo!(
            "Liveness::addVirtualAssignEnforced (Analyses/Liveness.h:124) — out of campaign scope"
        )
    }

    fn operand_to_index(&mut self, _value: Val) -> RegNode {
        todo!("Liveness::getOperandToIndex (Analyses/Liveness.h:143) — out of campaign scope")
    }
}

/// ONE UNIT'S `PropagationAnalysis::ExprInfo` AS THE SIMPLIFICATIONS READ IT — the four fields their
/// refusals test (`Analyses/PropagationAnalysis.h:99-102`), never the `AffineMap` itself, which is
/// MLIR affine machinery this crate does not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExprInfoShape {
    /// `propagated_args_.size()`.
    pub num_propagated_args: usize,
    /// `propagated_map_.getNumResults()`.
    pub num_map_results: usize,
    /// `propagated_map_.getNumDims()`.
    pub num_map_dims: usize,
    /// `cannot_be_resolved_`.
    pub cannot_be_resolved: bool,
}

/// `getFlattenedAffineExprs(propagated_map_, &flat_expr, &constraints)` ON SUCCESS — the coefficient
/// rows, and the one thing its callers ask the constraint system it also fills.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FlattenedAffineExprs {
    /// `flat_expr` — one row per result of the map, the dimension coefficients then the constant.
    pub rows: Vec<Vec<i64>>,
    /// `constraints.getNumLocalVars()` — ⛔ ANYTHING ABOVE ZERO IS DECLINED by every caller: MLIR
    /// cannot construct a local variable without an explicit representation.
    pub num_local_vars: usize,
}

/// THE `PropagationAnalysis&` A PASS IS HANDED — a trait for the same reason [`Liveness`] is one: the
/// analysis is not in this campaign, and a test must still be able to observe which values a ported
/// pass treats as carrying the same expression.
///
/// ⛔ `Analyses/PropagationAnalysis.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only
/// implementation is [`OutOfScopePropagationAnalysis`] and every method of it is a `todo!`. See
/// [`UnitIndexMap`] for the one part of this class a ported pass reaches through a narrower seam.
pub trait PropagationAnalysis {
    /// `areExpressionsSame(val1, val2)` (`Analyses/PropagationAnalysis.h:335`) — whether the two
    /// values' affine expressions agree unit by unit. ⭐ NOT `const` THERE (it memoises), so `&mut`.
    fn are_expressions_same(&mut self, val1: Val, val2: Val) -> bool;

    /// `getAffineExpression(val)` (`Analyses/PropagationAnalysis.h:308`) — the propagated affine
    /// expression of `val`, one `ExprInfo` per unit. ⭐ DEFAULTED TO A `todo!` like
    /// [`ExpressionEvaluator`]'s handle flavour, so a test double implements only what its unit asks.
    fn affine_expression(&mut self, val: Val) -> ExprInfoMap {
        let _ = val;
        todo!(
            "PropagationAnalysis::getAffineExpression (Analyses/PropagationAnalysis.h:308) — out of campaign scope"
        )
    }

    /// `ExprInfoMap::empty()` — no unit has an expression for this value, which is the staleness a
    /// simplification declines on rather than trusting.
    fn is_expr_info_map_empty(&mut self, map: ExprInfoMap) -> bool {
        let _ = map;
        todo!(
            "PropagationAnalysis::ExprInfoMap::empty (Analyses/PropagationAnalysis.h:106) — out of campaign scope"
        )
    }

    /// `ExprInfoMap::getExprInfoAt(at)->propagated_args_` — the SSA values the affine map at unit
    /// index `at` is written over. ⛔ `at == 0` IS NOT `getFirstExprInfo()`; that is
    /// [`PropagationAnalysis::first_propagated_args`].
    fn propagated_args(&mut self, map: ExprInfoMap, at: usize) -> Vec<Val> {
        let _ = (map, at);
        todo!(
            "PropagationAnalysis::ExprInfo::propagated_args_ (Analyses/PropagationAnalysis.h:100) — out of campaign scope"
        )
    }

    /// `ExprInfoMap::getFirstExprInfo()` (`Analyses/PropagationAnalysis.h:264-269`) REDUCED TO THE
    /// SAME FIELDS as [`PropagationAnalysis::expr_info_at`] — see [`ExprInfoShape`].
    ///
    /// ⛔ IT IS THE FIRST NON-NULL ENTRY OF `expr_info_list_`, NOT UNIT INDEX 0: a unit index goes
    /// through `getListIdxFromUnitIdx` (`Analyses/PropagationAnalysis.cpp:1722-1749`), and only
    /// `kIdentical` collapses it to 0 — which is why the reference has a second accessor at all, "for
    /// cases where we do not have the index list" (`Analyses/PropagationAnalysis.h:258-263`).
    fn first_expr_info(&mut self, map: ExprInfoMap) -> Option<ExprInfoShape> {
        let _ = map;
        todo!(
            "PropagationAnalysis::ExprInfoMap::getFirstExprInfo (Analyses/PropagationAnalysis.h:264) — out of campaign scope"
        )
    }

    /// `getFirstExprInfo()->propagated_args_` — the same list [`PropagationAnalysis::propagated_args`]
    /// answers, for the entry [`PropagationAnalysis::first_expr_info`] names.
    fn first_propagated_args(&mut self, map: ExprInfoMap) -> Vec<Val> {
        let _ = map;
        todo!(
            "PropagationAnalysis::ExprInfoMap::getFirstExprInfo (Analyses/PropagationAnalysis.h:264) — out of campaign scope"
        )
    }

    /// `getExprInfoAt(at)` (`Analyses/PropagationAnalysis.h:254`) REDUCED TO THE FIELDS A
    /// SIMPLIFICATION TESTS — see [`ExprInfoShape`]. ⭐ `None` IS THE `DT_CHECK_MSG(expr_info,
    /// "Expecting valid ExprInfo for unit")` its three callers open with.
    fn expr_info_at(&mut self, map: ExprInfoMap, at: usize) -> Option<ExprInfoShape> {
        let _ = (map, at);
        todo!(
            "PropagationAnalysis::ExprInfoMap::getExprInfoAt (Analyses/PropagationAnalysis.h:254) — out of campaign scope"
        )
    }

    /// `getFlattenedAffineExprs(getExprInfoAt(at)->propagated_map_, &flat_expr, &constraints)` —
    /// ⭐ `None` IS ITS `LogicalResult::failure()`, which every caller but e534 declines on.
    fn flattened_affine_exprs(
        &mut self,
        map: ExprInfoMap,
        at: usize,
    ) -> Option<FlattenedAffineExprs> {
        let _ = (map, at);
        todo!(
            "mlir::affine::getFlattenedAffineExpr + affine::FlatAffineValueConstraints \
             (mlir/Dialect/Affine/Analysis/AffineStructures.h) — MLIR upstream, out of campaign scope"
        )
    }

    /// `isPropagationSuccessful()` (`Analyses/PropagationAnalysis.h:313`) — whether the analysis ran
    /// to the end over this op. ⭐ `false` IS ITS `LogicalResult::failure()`, and e580 returns without
    /// walking anything on it.
    fn is_propagation_successful(&mut self) -> bool {
        todo!(
            "PropagationAnalysis::isPropagationSuccessful (Analyses/PropagationAnalysis.h:313) — out of campaign scope"
        )
    }

    /// `setExprInfoMapForValue(val, expr_map)` (`Analyses/PropagationAnalysis.h:323`) — the operand a
    /// simplification just created inherits the expression map of the op it stands for.
    fn set_expr_info_map_for_value(&mut self, val: Val, map: ExprInfoMap) {
        let _ = (val, map);
        todo!(
            "PropagationAnalysis::setExprInfoMapForValue (Analyses/PropagationAnalysis.h:323) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION: the affine expressions behind the answer are not ported.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopePropagationAnalysis;

impl PropagationAnalysis for OutOfScopePropagationAnalysis {
    fn are_expressions_same(&mut self, _val1: Val, _val2: Val) -> bool {
        todo!(
            "PropagationAnalysis::areExpressionsSame (Analyses/PropagationAnalysis.h:335) — out of campaign scope"
        )
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

/// ONE COLUMN OF A `TimeStamp` — `TimeStampColumnVal` (`Analyses/TimeStamps.h:39-91`) AS A PORTED
/// PASS READS IT: which of the class's own three `TimeStampColumnType`s this column is, and the op it
/// names.
///
/// ⛔ `Analyses/TimeStamps.{h,cpp}` IS OUT OF CAMPAIGN SCOPE and the analysis that BUILDS these
/// columns stays there. What is spelled here is exactly the four accessors a ported pass calls —
/// `isConstant()`, `isLoop()`/`getLoop()`, `isCond()`/`getCond()` (`:64-79`) — which read fields the
/// constructors set and compute nothing. `getConstantValue()`, `getLoopBound()` and
/// `getLoopBodyCycleCount()` are the analysis's own and no ported unit asks for them.
///
/// ⭐ AN ENUM BECAUSE `TimeStampColumnType` IS ONE, CLOSED AT THREE (`:40`): a column is a constant,
/// a loop iterator or a conditional, and `type_` is set in each constructor and never again.
/// ⛔ THE OP IS A POSITION, NOT `ForOp`/`IfOp`: `getLoop().getOperation()` is all
/// `e482_computeDependenciesSameBlock` does with one, and [`OpId`] is this crate's stand-in for the
/// pointer — so it goes stale exactly as [`Dependency`]'s ends do.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TimeStampColumnVal {
    /// `kConstant` — `explicit TimeStampColumnVal(int val)`.
    #[default]
    Constant,
    /// `kLoopIterator` — `explicit TimeStampColumnVal(sentient::ForOp op)`, at that op's position.
    Loop(OpId),
    /// `kConditional` — `explicit TimeStampColumnVal(sentient::IfOp op)`, at that op's position.
    Cond(OpId),
}

impl TimeStampColumnVal {
    /// `getLoop().getOperation()` when `isLoop()`, `getCond().getOperation()` when `isCond()` — the
    /// one question `e482_computeDependenciesSameBlock`'s parent-op scan asks of a column.
    #[must_use]
    pub fn region_op(&self) -> Option<&OpId> {
        match self {
            TimeStampColumnVal::Loop(op) | TimeStampColumnVal::Cond(op) => Some(op),
            TimeStampColumnVal::Constant => None,
        }
    }
}

/// WHICH WAY A CYCLE GAP RUNS — `enum directionType { forward, nextIter }` (`Analyses/TimeStamps.h:26`).
///
/// ⛔ BOTH ARMS ARE READ, AND THEY DECIDE DIFFERENT THINGS: `e482_computeDependenciesSameBlock`
/// records a hazard only for `forward` (`:169`), and `e483_computeDependenciesAcrossBlocks` uses
/// `forward` twice — once to SKIP the pairs e482 already banked and once to group by destination
/// (`:245`, `:259`). A `nextIter` gap survives into neither list unless `src == dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GapDirection {
    /// `forward` — `dst` reads in the same iteration `src` wrote in.
    Forward,
    /// `nextIter` — the read is a loop-carried one, `operationDistance` having come out non-positive.
    NextIter,
}

/// THE `TimeStamp &ts_analyzer` A PASS IS HANDED — a trait, for the reason
/// [`ExpressionEvaluator`] is one: the analysis behind it is not in this campaign and a test must
/// still be able to state its answers.
///
/// ⛔ `Analyses/TimeStamps.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only implementation is
/// [`OutOfScopeTimeStamps`] and every method of it is a `todo!` naming the analysis.
/// ⭐ THE FIVE METHODS ARE EXACTLY WHAT THE PORTED UNITS TOUCH: `time_stamps_op_order_`,
/// `time_stamps_[op]`, `isInSameBlock`, `getCyclesGap` and `reduceIntervals`. `intervalIntersection`,
/// `isIntervalEmpty`, `commonDimIndex`, `operationDistance`, `getNextIterCycleGap` and
/// `computeTimeStamps` are reached only from inside those and are not restated here.
pub trait TimeStamps {
    /// `time_stamps_op_order_` — the MACs the analysis timestamped, in the order it met them.
    fn op_order(&self) -> &[OpId];

    /// `time_stamps_[op]`. ⛔ A `std::map` SUBSCRIPT DEFAULT-CONSTRUCTS, so an op the analysis never
    /// timestamped answers the EMPTY column list rather than refusing.
    fn time_stamp(&self, op: &OpId) -> &[TimeStampColumnVal];

    /// `isInSameBlock` (`Analyses/TimeStamps.cpp:287-296`) — do two timestamps differ only in their
    /// last column.
    fn is_in_same_block(&self, src: &[TimeStampColumnVal], dst: &[TimeStampColumnVal]) -> bool;

    /// `getCyclesGap` (`Analyses/TimeStamps.cpp:137-163`) — the smaller of the direct and
    /// next-iteration distances, and which one it was.
    fn cycles_gap(&mut self, src: &OpId, dst: &OpId) -> (Cycles, GapDirection);

    /// `reduceIntervals` (`Analyses/TimeStamps.cpp:228-254`) — collapse a hazard list in place onto
    /// the fewest intervals whose intersections still cover every hazard in it.
    fn reduce_intervals(&mut self, intervals: &mut Vec<Dependency>);

    /// `int computeTimeStamps(Operation *op, std::vector<TimeStampColumnVal> &parent_time_stamps)`
    /// (`Analyses/TimeStamps.h:139`) — timestamps `op` and everything under it, appending its own
    /// columns to `parent_time_stamps`.
    ///
    /// ⭐ `op` IS A BODY HERE because every call in this campaign passes the analyzer's own
    /// `root_` — the `dataflow.program_unit` it was constructed with (`Analyses/TimeStamps.h:100`,
    /// `:112`) — and the returned cycle count is discarded at that call.
    fn compute_time_stamps(&mut self, op: &[Op], parent_time_stamps: &mut Vec<TimeStampColumnVal>);
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeTimeStamps;

impl TimeStamps for OutOfScopeTimeStamps {
    fn op_order(&self) -> &[OpId] {
        todo!(
            "TimeStamp::time_stamps_op_order_ (Analyses/TimeStamps.h:104) — out of campaign scope"
        )
    }

    fn time_stamp(&self, _op: &OpId) -> &[TimeStampColumnVal] {
        todo!("TimeStamp::time_stamps_ (Analyses/TimeStamps.h:103) — out of campaign scope")
    }

    fn is_in_same_block(&self, _src: &[TimeStampColumnVal], _dst: &[TimeStampColumnVal]) -> bool {
        todo!("TimeStamp::isInSameBlock (Analyses/TimeStamps.cpp:287) — out of campaign scope")
    }

    fn cycles_gap(&mut self, _src: &OpId, _dst: &OpId) -> (Cycles, GapDirection) {
        todo!("TimeStamp::getCyclesGap (Analyses/TimeStamps.cpp:137) — out of campaign scope")
    }

    fn reduce_intervals(&mut self, _intervals: &mut Vec<Dependency>) {
        todo!("TimeStamp::reduceIntervals (Analyses/TimeStamps.cpp:228) — out of campaign scope")
    }

    fn compute_time_stamps(
        &mut self,
        _op: &[Op],
        _parent_time_stamps: &mut Vec<TimeStampColumnVal>,
    ) {
        todo!("TimeStamp::computeTimeStamps (Analyses/TimeStamps.h:139) — out of campaign scope")
    }
}

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

/// WHICH UNIT A REGISTER-PRESSURE ANSWER IS IN — `RegisterPressure::Metric`
/// (`Analyses/RegisterPressureAnalysis.h:37-43`), and the reason [`Pressure`] carries no unit of its
/// own: the same number means registers, free registers or a percentage depending on this.
///
/// ⭐ `kDefault` IS `kNumRegisters`, an alias and not a fifth case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    /// `kNumRegisters`, which `kDefault` names.
    #[default]
    NumRegisters,
    /// `kNumFreeRegisters`.
    NumFreeRegisters,
    /// `kPercentUtilization`.
    PercentUtilization,
    /// `kPercentFree`.
    PercentFree,
}

/// ONE REGISTER-PRESSURE ANSWER — the `unsigned` of `getOrComputeRegisterPressure`, read in whichever
/// unit the [`Metric`] asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pressure(pub u32);

/// THE `RegisterPressure&` A PASS IS HANDED — a trait for the same reason [`Liveness`] is one: the
/// estimate is not in this campaign, and a test must still be able to observe what a pass asks of it.
///
/// ⛔ `Analyses/RegisterPressureAnalysis.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only
/// implementation is [`OutOfScopeRegisterPressure`] and every method of it is a `todo!`.
/// ⭐ ONE TRAIT FOR BOTH `RegisterPressure` AND `RegisterPressureAnalysis`: the latter is a pass-manager
/// wrapper that forwards all three methods to a `RegisterPressure` member (`:156-187`).
pub trait RegisterPressure {
    /// `getOrComputeRegisterPressure(locale, metric)` (`:71`) — the estimate for one register file.
    fn get_or_compute_register_pressure(&mut self, locale: RegType, metric: Metric) -> Pressure;

    /// `computeRegisterPressureForAllLocales()` (`:78`) — *"meant to be used for debugging and
    /// information collection purposes only"*, which is why every caller of it is a debug block.
    fn compute_register_pressure_for_all_locales(&mut self);

    /// `dump()` (`:80`) — a `String` rather than a write to `llvm::dbgs()`, as the ported dumps of
    /// [`super::scalar_copy_insertion_for_symbols`] are.
    fn dump(&self) -> String;
}

/// THE ONE CRATE IMPLEMENTATION: the estimate is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeRegisterPressure;

impl RegisterPressure for OutOfScopeRegisterPressure {
    fn get_or_compute_register_pressure(&mut self, _locale: RegType, _metric: Metric) -> Pressure {
        todo!(
            "RegisterPressure::getOrComputeRegisterPressure \
             (Analyses/RegisterPressureAnalysis.h:71) — out of campaign scope"
        )
    }

    fn compute_register_pressure_for_all_locales(&mut self) {
        todo!(
            "RegisterPressure::computeRegisterPressureForAllLocales \
             (Analyses/RegisterPressureAnalysis.h:78) — out of campaign scope"
        )
    }

    fn dump(&self) -> String {
        todo!(
            "RegisterPressure::dump (Analyses/RegisterPressureAnalysis.h:80) — out of campaign scope"
        )
    }
}
