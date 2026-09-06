// SPDX-License-Identifier: Apache-2.0
//! WHICH PRECISION EACH OPERAND, RESULT AND COMPUTE CARRIES.
//!
//! Ported from `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30-89`:
//! `isSentientBinaryLogicalOp`, both `getInputPrecisionFromOperand` overloads,
//! `getResultPrecisionFromOperands` and `getComputePrecisionOfOp`.
//!
//! ⭐⭐ THREE PRECISIONS PER COMPUTE, AND THEY COME FROM THREE DIFFERENT PLACES. The `.td` declares
//! `opAPrecision`, `ResultPrecision` and `ComputePrecision` as separate attributes
//! (`SentientOps.td:232-288`), and the reference fills each from its own source — so a lowering that
//! computed one and reused it for the others would be wrong three ways at once:
//!
//! | attribute | source |
//! |---|---|
//! | `op<X>Precision` | that operand's **original** precision (`getInputPrecisionFromOperand`) |
//! | `ResultPrecision` | the **first present** operand's **on-the-fly conversion** precision |
//! | `ComputePrecision` | the OP's own element type, remapped (`getComputePrecisionOfOp`) |
//!
//! ⛔⛔ SO AN OPERAND CARRIES **TWO** PRECISIONS WHILE THE LOWERING RUNS. `VectorOperand` holds
//! `orig_precision_` and `on_the_fly_conv_precision_` side by side
//! (`VectorOperands.hpp:67-68`) — what the operand *was*, and what a conversion on the way in made it.
//! Only the first reaches `op<X>Precision`; only the second reaches `ResultPrecision`, and only from
//! whichever operand is present first. Collapsing them into one field loses the result precision.
//!
//! ⭐ THE ISLAND IS UNAFFECTED, AND THAT IS THE RIGHT DIVISION. `sentient::Operand` carries one
//! `precision`, because one is what the *emitted attribute* holds. The pair is a fact about lowering,
//! so it lives here in the bridge and never reaches the island.

use crate::islands::sentient::dialects::sentient::{BinaryOp, Precision};

/// AN OPERAND'S TWO PRECISIONS WHILE THE LOWERING RUNS.
///
/// ⛔ BOTH NAMED, NEITHER DEFAULTED. `VectorOperands.hpp:67-68` declares them adjacent and both as
/// `std::string`, which is exactly the transposition a newtype rule exists to prevent: swapping them
/// yields a compute whose operand attribute claims the converted width and whose result attribute
/// claims the original.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandPrecision {
    /// `orig_precision_` — what the operand was before any conversion on the way in.
    ///
    /// ⭐ THIS IS WHAT `op<X>Precision` GETS.
    pub original: Precision,
    /// `on_the_fly_conv_precision_` — what a conversion on the way in made it.
    ///
    /// ⭐ THIS IS WHAT `ResultPrecision` GETS, and only from the first present operand.
    pub converted: Precision,
}

impl OperandPrecision {
    /// `getInputPrecisionFromOperand` (`VectorChainHelper.cpp:36-41`) — the operand attribute's value.
    ///
    /// ⭐ THE `fp80 -> fp8` REMAP IS THE WHOLE BODY, and the reference states why in a comment it
    /// repeats three times: *"Currently, we use fp80 type in MLIR to represent fp8."* MLIR has no fp8
    /// type, so the eighty-bit float stands in for it. [`Precision::of_mlir_element`] is the one place
    /// that substitution is undone.
    #[must_use]
    pub const fn for_operand_attribute(self) -> Precision {
        self.original
    }
}

/// THE RESULT PRECISION OF A COMPUTE — `getResultPrecisionFromOperands`
/// (`VectorChainHelper.cpp:52-62`).
///
/// ⛔⛔ THE **FIRST PRESENT** OPERAND DECIDES IT, AND THE LOOP RETURNS ON THAT ONE. The reference walks
/// the operand list and returns inside the first `has_value()` arm — it does not consult the rest, and
/// it does not check that they agree. So this is a *first-wins* rule, not a consensus, and an
/// implementation that scanned all operands looking for a common precision would differ on any compute
/// whose operands disagree.
///
/// ⭐ AND IT READS THE **CONVERTED** PRECISION, not the original — the one field
/// [`OperandPrecision::for_operand_attribute`] does not return.
///
/// ⛔ `None` WHERE NO OPERAND IS PRESENT. The reference returns `""`, an empty string it later writes
/// into an attribute; here absence is an `Option` so it cannot be printed as a precision. This is an
/// ABSENCE, not a refusal — there is no error path.
#[must_use]
pub fn result_precision(operands: &[Option<OperandPrecision>]) -> Option<Precision> {
    operands
        .iter()
        .flatten()
        .next()
        .map(|operand| operand.converted)
}

/// WHETHER A BINARY OPERATOR IS A LOGICAL ONE — `isSentientBinaryLogicalOp`
/// (`VectorChainHelper.cpp:30-34`).
///
/// ⛔ FOUR OPERATORS, AND THE SPELLINGS ARE THE `.td`'S: `and0`, `or0`, `xnor`, `and_not`. The first
/// two are not `and`/`or` because those are C++ keywords — the `.td` says so in as many words — so a
/// reader completing the set from intuition writes attributes the parser rejects.
///
/// ⛔ EXHAUSTIVE, NO WILDCARD. A fifty-ninth operator must state whether it is logical rather than
/// inherit `false`.
#[must_use]
pub const fn is_logical(op: BinaryOp) -> bool {
    match op {
        BinaryOp::And | BinaryOp::Or | BinaryOp::Xnor | BinaryOp::AndNot => true,
        BinaryOp::Min
        | BinaryOp::Max
        | BinaryOp::AbsMin
        | BinaryOp::AbsMax
        | BinaryOp::Add
        | BinaryOp::Mul
        | BinaryOp::Sub
        | BinaryOp::MulDiv2
        | BinaryOp::GcvtImm(_)
        | BinaryOp::FcvtImm(_)
        | BinaryOp::Merge { .. }
        | BinaryOp::Pack(_)
        | BinaryOp::CompareNeq
        | BinaryOp::CompareEq
        | BinaryOp::CompareLt
        | BinaryOp::CompareLe => false,
    }
}

impl Precision {
    /// WHAT AN MLIR ELEMENT TYPE MEANS AS A SENTIENT PRECISION.
    ///
    /// ⛔⛔ `fp80` IS **fp8**, AND THAT IS NOT A TYPO. MLIR has no fp8 type, so the reference uses the
    /// eighty-bit float to stand for it — *"Currently, we use fp80 type in MLIR to represent fp8"*,
    /// a comment `VectorChainHelper.cpp` repeats at `:39`, `:58` and `:70` because the substitution has
    /// to be undone at every one of the three precision sources. Reading `fp80` literally gives an
    /// eighty-bit compute on a machine whose widest float is 32.
    #[must_use]
    pub const fn of_mlir_element(element: MlirElement) -> Precision {
        match element {
            // ⛔ THE STAND-IN, UNDONE.
            MlirElement::Fp80 => Precision::Fp8,
            MlirElement::Fp16 => Precision::Fp16,
            MlirElement::Bf16 => Precision::Bf16,
            MlirElement::DlFp16 => Precision::IeeeFp16,
            MlirElement::Fp32 => Precision::Fp32,
            MlirElement::Int8 => Precision::Int8,
            MlirElement::Int16 => Precision::Int16,
            MlirElement::Int32 => Precision::Int32,
        }
    }

    /// THE COMPUTE PRECISION OF AN OP — `getComputePrecisionOfOp` (`VectorChainHelper.cpp:65-77`).
    ///
    /// ⛔⛔ **`bf16` AND `dlfp16` COMPUTE AS `fp16`, AND ONLY HERE.** The reference remaps them in
    /// `getComputePrecisionOfOp` and in neither of the operand nor result paths — its comment says
    /// *"bf16, dlpfp16 correspond to fp16 compute precision"*. So the SAME element type yields `bf16`
    /// in `opAPrecision` and `fp16` in `ComputePrecision`, deliberately. A single shared conversion
    /// would make those agree and be wrong on one of them.
    ///
    /// ⛔ AND A PACK OR MERGE IS ALWAYS `fp16`, WHATEVER ITS ELEMENTS. The reference returns early for
    /// both before looking at the element type at all (`:67`) — they are lane rearrangements, so the
    /// element width is not what the datapath computes in.
    #[must_use]
    pub const fn compute_of(op: BinaryOp, element: MlirElement) -> Precision {
        // ⛔ THE EARLY RETURN IS LOAD-BEARING: it precedes the element-type read.
        if matches!(op, BinaryOp::Pack(_) | BinaryOp::Merge { .. }) {
            return Precision::Fp16;
        }
        match Precision::of_mlir_element(element) {
            // ⛔ COMPUTE ONLY — see this function's note.
            Precision::Bf16 | Precision::IeeeFp16 => Precision::Fp16,
            other => other,
        }
    }
}

/// AN MLIR ELEMENT TYPE, AS THE VECTORCHAIN OPS CARRY IT.
///
/// ⛔ A SEPARATE TYPE FROM [`Precision`] BECAUSE THE TWO ARE NOT THE SAME SET, and conflating them is
/// what the `fp80` stand-in punishes. `fp80` is an MLIR element type and NOT a sentient precision;
/// `fp8` is a sentient precision with no MLIR element type. The mapping between them is
/// [`Precision::of_mlir_element`] and it is not a bijection.
///
/// ⛔ NOT A STRING. The reference threads these as `std::string` through `getPrecisionInString`, and
/// compares them with `==` against literals at three sites — which is how `"fp80"` came to be tested
/// for three separate times. A closed set is an enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlirElement {
    /// `f80` — ⛔ MEANS fp8; see [`Precision::of_mlir_element`].
    Fp80,
    /// `f16`.
    Fp16,
    /// `bf16`.
    Bf16,
    /// `dlfp16` — the deep-learning fp16 the reference names beside `bf16`.
    DlFp16,
    /// `f32`.
    Fp32,
    /// `i8`.
    Int8,
    /// `i16`.
    Int16,
    /// `i32`.
    Int32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 🎯 THE THREE PRECISION SOURCES DISAGREE ON PURPOSE, for one element type.
    ///
    /// ⭐ THIS IS THE WHOLE POINT OF THE FILE. A `bf16` operand reports `bf16` as its operand
    /// attribute and `fp16` as the compute precision. If a refactor ever makes these agree, it has
    /// collapsed three sources into one and this fails.
    #[test]
    fn bf16_computes_as_fp16_but_reports_as_bf16() {
        let operand = OperandPrecision {
            original: Precision::of_mlir_element(MlirElement::Bf16),
            converted: Precision::of_mlir_element(MlirElement::Bf16),
        };
        assert_eq!(operand.for_operand_attribute(), Precision::Bf16);
        assert_eq!(
            Precision::compute_of(BinaryOp::Add, MlirElement::Bf16),
            Precision::Fp16
        );
    }

    /// 🎯 `fp80` IS UNDONE AT EVERY SOURCE, because the reference undoes it at all three.
    #[test]
    fn fp80_is_fp8_everywhere() {
        assert_eq!(Precision::of_mlir_element(MlirElement::Fp80), Precision::Fp8);
        assert_eq!(
            Precision::compute_of(BinaryOp::Mul, MlirElement::Fp80),
            Precision::Fp8
        );
        let operand = OperandPrecision {
            original: Precision::of_mlir_element(MlirElement::Fp80),
            converted: Precision::of_mlir_element(MlirElement::Fp80),
        };
        assert_eq!(operand.for_operand_attribute(), Precision::Fp8);
    }

    /// 🎯 A PACK OR MERGE IGNORES ITS ELEMENT TYPE — the early return precedes the element read.
    #[test]
    fn a_lane_rearrangement_always_computes_in_fp16() {
        for element in [MlirElement::Int8, MlirElement::Fp32, MlirElement::Fp80] {
            assert_eq!(
                Precision::compute_of(BinaryOp::Pack(crate::islands::sentient::dialects::sentient::PackIndex::P0), element),
                Precision::Fp16,
                "a pack computes in fp16 whatever its elements"
            );
        }
    }

    /// 🎯 THE RESULT PRECISION IS FIRST-WINS, NOT CONSENSUS.
    ///
    /// ⛔ THE REFERENCE RETURNS INSIDE THE FIRST `has_value()` ARM and never consults the rest. Two
    /// operands that disagree therefore yield the FIRST one's converted precision, and an
    /// implementation that looked for agreement would differ here.
    #[test]
    fn the_result_precision_is_the_first_present_operands() {
        let fp8 = OperandPrecision {
            original: Precision::Fp16,
            converted: Precision::Fp8,
        };
        let fp32 = OperandPrecision {
            original: Precision::Fp16,
            converted: Precision::Fp32,
        };
        // absent first, so the fp8 operand wins — and note it is the CONVERTED field that is read,
        // not `original`, which is fp16 on both.
        assert_eq!(
            result_precision(&[None, Some(fp8), Some(fp32)]),
            Some(Precision::Fp8)
        );
        // reversed: now fp32 wins, proving the rule is positional rather than a preference.
        assert_eq!(
            result_precision(&[Some(fp32), Some(fp8)]),
            Some(Precision::Fp32)
        );
        // ⭐ NO OPERAND PRESENT IS AN ABSENCE, NOT A REFUSAL. The reference writes `""`.
        assert_eq!(result_precision(&[None, None]), None);
    }

    /// 🎯 THE FOUR LOGICAL OPERATORS, AND THE SPELLINGS THAT TRIP PEOPLE UP.
    #[test]
    fn only_four_operators_are_logical() {
        for op in [BinaryOp::And, BinaryOp::Or, BinaryOp::Xnor, BinaryOp::AndNot] {
            assert!(is_logical(op), "{op:?} is one of the four");
        }
        for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Min, BinaryOp::CompareEq] {
            assert!(!is_logical(op), "{op:?} is not logical");
        }
        // ⛔ AND THE SPELLINGS ARE `and0`/`or0`, not `and`/`or` — C++ keywords forced it.
        assert_eq!(BinaryOp::And.spelling(), "and0");
        assert_eq!(BinaryOp::Or.spelling(), "or0");
    }
}
