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
//! | `ComputePrecision` | the op's own format, remapped (`getComputePrecisionOfOp`) |
//!
//! ⛔⛔ SO AN OPERAND CARRIES **TWO** PRECISIONS WHILE THE LOWERING RUNS. `VectorOperand` holds
//! `orig_precision_` and `on_the_fly_conv_precision_` side by side (`VectorOperands.hpp:67-68`) — what
//! the operand *was*, and what a conversion on the way in made it. Only the first reaches
//! `op<X>Precision`; only the second reaches `ResultPrecision`, and only from whichever operand is
//! present first. Collapsing them into one field loses the result precision.
//!
//! ⭐ THE ISLAND IS UNAFFECTED, AND THAT IS THE RIGHT DIVISION. `sentient::Operand` carries one
//! `precision`, because one is what the *emitted attribute* holds. The pair is a fact about lowering.
//!
//! # 🛑 WHAT THIS FILE DELIBERATELY DOES NOT PORT: THE `fp80` STAND-IN
//!
//! ⛔⛔ AN EARLIER VERSION OF THIS FILE PORTED IT, AND IT WAS A WORKAROUND MASQUERADING AS A RULE.
//! `VectorChainHelper.cpp` remaps `fp80 -> fp8` at `:39`, `:58` and `:70`, each under the comment
//! *"Currently, we use fp80 type in MLIR to represent fp8"* — the eighty-bit float standing in for a
//! type their MLIR lacked. The repetition made it look load-bearing; it is the opposite.
//!
//! ⭐ WE HAVE NO SUCH PROBLEM, so applying that remap would be WRONG rather than merely redundant.
//! [`crate::islands::dataflow_ir::ty::ElemType`] names `F8E4M3Fn` — a real fp8 element type, which is
//! what the island's printer emits — and the tape's formats are the generated [`DataType`], which spells
//! three distinct fp8s outright. There is no eighty-bit float anywhere in our vocabulary to undo.
//!
//! ⛔ AND THE FIRST VERSION INVENTED AN `MlirElement` ENUM TO HOLD IT — an MLIR-shaped type in a crate
//! that deliberately has no MLIR. That is precisely the "port the rule, never the plumbing" failure:
//! the C++ reads an element type off an in-memory `mlir::Type`, and transcribing that reading gave us a
//! type whose only distinctive member (`fp80`) does not exist for us. The mapping below starts from
//! [`DataType`], the closed generated enum the tape already carries, which is also what
//! `ElemType::of` starts from.

use crate::generated::DataType;
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
    /// ⭐ ONCE THE `fp80` REMAP IS DROPPED THE REFERENCE'S BODY IS A FIELD READ, which is why this is
    /// a getter rather than a conversion. Keeping it named says which of the two fields the operand
    /// attribute takes — the thing the reference's two same-named overloads obscure.
    #[must_use]
    pub const fn for_operand_attribute(self) -> Precision {
        self.original
    }

    /// AN OPERAND WHOSE FORMAT NO CONVERSION CHANGED — both precisions the same.
    #[must_use]
    pub const fn unconverted(format: DataType) -> OperandPrecision {
        let precision = Precision::of_format(format);
        OperandPrecision {
            original: precision,
            converted: precision,
        }
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
/// into an attribute; here absence is an `Option` so it cannot be printed as a precision. An ABSENCE,
/// not a refusal — there is no error path in this bridge.
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
/// ⛔ EXHAUSTIVE, NO WILDCARD. A new operator must state whether it is logical rather than inherit
/// `false`.
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
    /// WHAT A TAPE FORMAT IS, AS A SENTIENT PRECISION.
    ///
    /// ⭐⭐ TOTAL OVER THE GENERATED [`DataType`], which is the same footing
    /// [`crate::islands::dataflow_ir::ty::ElemType::of`] stands on: a template introducing a format
    /// stops this compiling rather than falling through. That is why the mapping starts here and not
    /// from an element type — an element type carries a bare `Int(u32)`, and a width the ISA does not
    /// define would have nowhere to go.
    ///
    /// ⛔ THREE fp8s MAP TO ONE `fp8`. `Sen143Fp8`, `Sen080Fp8` and `Sen053Fp8` are three different
    /// exponent/mantissa splits (`sendefs.cpp:129-141`) and the ISA has ONE `fp8` precision, so the
    /// split is carried by the format and not by this attribute. Losing it here is correct; inventing
    /// three precisions to preserve it would emit attributes the parser does not know.
    ///
    /// ⛔ AND `Senint24` IS SIXTEEN BITS WIDE but its own precision — `int24` is a real enumerator
    /// (`SentientTypes.td:54`) even though [`DataType::bits`] answers 16 for it. Width and precision
    /// are different questions about one format.
    #[must_use]
    pub const fn of_format(format: DataType) -> Precision {
        match format {
            DataType::Sen169Fp16 => Precision::Fp16,
            DataType::Bfloat16 => Precision::Bf16,
            DataType::IeeeFp32 => Precision::Fp32,
            // ⛔ THREE SPLITS, ONE PRECISION — see this function's note.
            DataType::Sen143Fp8 | DataType::Sen080Fp8 | DataType::Sen053Fp8 => Precision::Fp8,
            DataType::Sen121Fp4 => Precision::Fp4,
            DataType::Senint4 => Precision::Int4,
            DataType::Senint8 => Precision::Int8,
            // ⛔ ITS OWN PRECISION, THOUGH IT OCCUPIES SIXTEEN BITS.
            DataType::Senint24 => Precision::Int24,
            DataType::Senuint32 => Precision::Int32,
            // ⭐ A PREDICATE, NOT A NUMBER. A mask element is one bit.
            DataType::Bool => Precision::Int1,
        }
    }

    /// THE COMPUTE PRECISION OF A BINARY OP — `getComputePrecisionOfOp`
    /// (`VectorChainHelper.cpp:65-77`).
    ///
    /// ⛔⛔ **`bf16` COMPUTES AS `fp16`, AND ONLY HERE.** The reference remaps it in
    /// `getComputePrecisionOfOp` and in neither the operand nor the result path — its comment is
    /// *"bf16, dlpfp16 correspond to fp16 compute precision"*. So the SAME format yields `bf16` in
    /// `opAPrecision` and `fp16` in `ComputePrecision`, deliberately. A single shared conversion would
    /// make those agree and be wrong on one of them.
    ///
    /// ⭐ THE REFERENCE ALSO REMAPS `dlfp16`, WHICH WE CANNOT EXPRESS AND SO DO NOT WRITE. There is no
    /// `dlfp16` in [`DataType`], so that arm has no input here — an absent arm rather than a dead one,
    /// which is the difference between a type that cannot say a thing and a match that ignores it.
    ///
    /// ⛔ AND A PACK OR MERGE IS ALWAYS `fp16`, WHATEVER ITS FORMAT. The reference returns early for
    /// both before reading the element type at all (`:67`) — they are lane rearrangements, so the
    /// element width is not what the datapath computes in.
    #[must_use]
    pub const fn compute_of(op: BinaryOp, format: DataType) -> Precision {
        // ⛔ THE EARLY RETURN IS LOAD-BEARING: it precedes the format read.
        if matches!(op, BinaryOp::Pack(_) | BinaryOp::Merge { .. }) {
            return Precision::Fp16;
        }
        match Precision::of_format(format) {
            // ⛔ COMPUTE ONLY — see this function's note.
            Precision::Bf16 => Precision::Fp16,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::islands::sentient::dialects::sentient::PackIndex;

    /// 🎯 THE THREE PRECISION SOURCES DISAGREE ON PURPOSE, for one format.
    ///
    /// ⭐ THIS IS THE WHOLE POINT OF THE FILE. A `bf16` operand reports `bf16` as its operand
    /// attribute and `fp16` as the compute precision. If a refactor ever makes these agree, it has
    /// collapsed three sources into one and this fails.
    #[test]
    fn bf16_computes_as_fp16_but_reports_as_bf16() {
        let operand = OperandPrecision::unconverted(DataType::Bfloat16);
        assert_eq!(operand.for_operand_attribute(), Precision::Bf16);
        assert_eq!(
            Precision::compute_of(BinaryOp::Add, DataType::Bfloat16),
            Precision::Fp16
        );
    }

    /// 🎯 OUR fp8 IS fp8 AT EVERY SOURCE — no stand-in to undo.
    ///
    /// ⛔ THE POINT IS THE ABSENCE OF A REMAP. The reference turns `fp80` into `fp8` three times
    /// because its MLIR had no fp8 type; our formats name three fp8 splits outright, so a value that
    /// enters as fp8 stays fp8 through all three sources.
    #[test]
    fn every_fp8_split_is_fp8_at_all_three_sources() {
        for format in [DataType::Sen143Fp8, DataType::Sen080Fp8, DataType::Sen053Fp8] {
            assert_eq!(Precision::of_format(format), Precision::Fp8);
            assert_eq!(Precision::compute_of(BinaryOp::Mul, format), Precision::Fp8);
            assert_eq!(
                OperandPrecision::unconverted(format).for_operand_attribute(),
                Precision::Fp8
            );
        }
    }

    /// 🎯 A PACK OR MERGE IGNORES ITS FORMAT — the early return precedes the format read.
    #[test]
    fn a_lane_rearrangement_always_computes_in_fp16() {
        for format in [DataType::Senint8, DataType::IeeeFp32, DataType::Sen143Fp8] {
            assert_eq!(
                Precision::compute_of(BinaryOp::Pack(PackIndex::P0), format),
                Precision::Fp16,
                "a pack computes in fp16 whatever its format"
            );
            assert_eq!(
                Precision::compute_of(
                    BinaryOp::Merge {
                        width: crate::islands::sentient::dialects::sentient::MergeWidth::W16,
                        high: false
                    },
                    format
                ),
                Precision::Fp16
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
        // ⭐ ORIGINAL AND CONVERTED DELIBERATELY DIFFER, so the test can tell which field is read.
        let to_fp8 = OperandPrecision {
            original: Precision::Fp16,
            converted: Precision::Fp8,
        };
        let to_fp32 = OperandPrecision {
            original: Precision::Fp16,
            converted: Precision::Fp32,
        };
        assert_eq!(
            result_precision(&[None, Some(to_fp8), Some(to_fp32)]),
            Some(Precision::Fp8),
            "the first PRESENT operand decides, and its CONVERTED field is read"
        );
        assert_eq!(
            result_precision(&[Some(to_fp32), Some(to_fp8)]),
            Some(Precision::Fp32),
            "reversed — so the rule is positional, not a preference between precisions"
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
        for op in [
            BinaryOp::Add,
            BinaryOp::Mul,
            BinaryOp::Min,
            BinaryOp::CompareEq,
        ] {
            assert!(!is_logical(op), "{op:?} is not logical");
        }
        // ⛔ AND THE SPELLINGS ARE `and0`/`or0`, not `and`/`or` — C++ keywords forced it.
        assert_eq!(BinaryOp::And.spelling(), "and0");
        assert_eq!(BinaryOp::Or.spelling(), "or0");
    }

    /// 🎯 EVERY FORMAT MAPS, because the match is total over the generated enum.
    ///
    /// ⭐ THE TEST IS THE COMPILER'S, not this body's: a new `DataType` variant fails to build
    /// `of_format`. This walks the ones that exist to pin the two that surprise — `Senint24`, whose
    /// precision is `int24` though it occupies sixteen bits, and `Bool`, which is one bit.
    #[test]
    fn the_two_surprising_formats() {
        assert_eq!(Precision::of_format(DataType::Senint24), Precision::Int24);
        assert_eq!(DataType::Senint24.bits().0, 16, "sixteen bits wide, int24 precision");
        assert_eq!(Precision::of_format(DataType::Bool), Precision::Int1);
    }
}
