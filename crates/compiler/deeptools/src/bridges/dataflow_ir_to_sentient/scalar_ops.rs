// SPDX-License-Identifier: Apache-2.0
//! WHICH `sentient` OP AN UPSTREAM `arith` OR `scf` OP BECOMES.
//!
//! Ported from `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp` (`LowerAddIOpToSentient`
//! `:79`, `LowerSubIOpToSentient` `:90`, `LowerMulIOpToSentient` `:102`,
//! `LowerConstantIndexToSentient` `:347`, `LowerConstantIntToSentient` `:358`,
//! `getSentientCmpIPredicate` `:36`) and `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp`
//! (`getSentientCmpIPredicate` `:33`).
//!
//! # 🛑 EACH `Lower*ToSentient` IS THREE LINES OF PLUMBING AROUND ONE FACT
//!
//! ⛔ THE BODIES ARE ALL THE SAME SHAPE: build the sentient op, `replaceAllUsesWith`, `erase`. That is
//! rewriting an in-memory module — we emit, so none of it is owed. **The fact is the correspondence**,
//! and it is the only thing here worth carrying:
//!
//! | upstream | sentient |
//! |---|---|
//! | `arith.addi` | `sentient.scalar_add` |
//! | `arith.subi` | `sentient.scalar_sub` |
//! | `arith.muli` | `sentient.scalar_mul` |
//! | `arith.constant` (index or int) | `sentient.scalar_constant` |
//!
//! ⭐ AND ONE ASYMMETRY WORTH KEEPING: `LowerMulIOpToSentient` copies the source op's attribute
//! dictionary (`setAttrs`, `:108`) while the add and sub paths have that line **commented out** with
//! *"We do not use Arith Attributes"* (`:96`). So mul propagates attributes and add/sub deliberately do
//! not. Whether that is intent or an oversight in the reference, it is a difference our emission has to
//! decide about rather than average over — [`Scalar::propagates_attributes`] records it.
//!
//! ⛔⛔ `getSentientCmpIPredicate` IS DUPLICATED IN BOTH FILES, and the two copies differ only in how
//! they fail: `StandardToSentient.cpp:52` does `DT_CHECK(0)` then returns `eq` *"to silence to
//! warning"*, while `SCFToSentient.cpp:48` calls `llvm_unreachable`. Two spellings of one mapping is
//! how the two drift; there is one here, and it is total, so neither failure arm exists.

use crate::islands::dataflow_ir::dialects::arith;
use crate::islands::sentient::dialects::sentient::CmpPredicate;

/// WHICH SCALAR `sentient` OP AN `arith` OP BECOMES.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    /// `arith.addi` -> `sentient.scalar_add`.
    Add,
    /// `arith.subi` -> `sentient.scalar_sub`.
    Sub,
    /// `arith.muli` -> `sentient.scalar_mul`.
    Mul,
    /// `arith.constant` -> `sentient.scalar_constant`, index and integer alike.
    ///
    /// ⭐ ONE ARM FOR BOTH. The reference has `LowerConstantIndexToSentient` and
    /// `LowerConstantIntToSentient` as separate functions, but they build the same
    /// `sentient::ConstantOp` — they differ only in how they read the value out of MLIR
    /// (`.value()` against a `BoolAttr` check), which is the plumbing half.
    Constant,
}

impl Scalar {
    /// WHETHER THE LOWERING CARRIES THE SOURCE OP'S ATTRIBUTES ACROSS.
    ///
    /// ⛔⛔ ONLY `mul` DOES, AND THAT IS THE REFERENCE'S BEHAVIOUR RATHER THAN A TIDY RULE.
    /// `LowerMulIOpToSentient` calls `setAttrs(muli_op->getAttrDictionary())` (`:108`); the same line
    /// in the add and sub paths is commented out above the words *"We do not use Arith Attributes"*
    /// (`:85`, `:96`). Recorded rather than smoothed, because smoothing it in either direction changes
    /// what we emit relative to the reference and the golden diff would then disagree for a reason
    /// nobody could place.
    #[must_use]
    pub const fn propagates_attributes(self) -> bool {
        match self {
            Scalar::Mul => true,
            Scalar::Add | Scalar::Sub | Scalar::Constant => false,
        }
    }
}

/// WHAT AN `arith` OP LOWERS TO — total over the ops our island can express.
///
/// ⛔ `None` FOR THE ONES THAT ARE NOT SCALAR LOWERINGS, and that is a classification rather than a
/// failure: `arith.cmpi` becomes a predicate on a `sentient.if` (see [`predicate`]), the connectives
/// become mask logic, and a dense constant is a vector rather than a scalar. Each has its own path, so
/// answering "not a scalar op" here is the right answer and not an inability.
#[must_use]
pub const fn scalar_for(op: &arith::Op) -> Option<Scalar> {
    match op {
        arith::Op::Constant { .. } => Some(Scalar::Constant),
        // ⛔ NOT SCALAR LOWERINGS — see this function's note.
        arith::Op::True { .. }
        | arith::Op::Compare { .. }
        | arith::Op::Logic { .. }
        | arith::Op::DenseConstant { .. } => None,
    }
}

/// AN `arith` COMPARISON PREDICATE AS A SENTIENT ONE — `getSentientCmpIPredicate`.
///
/// ⭐ SIX CASES AND A TOTAL MAPPING. The reference's `if`/`else if` chain ends in a failure arm it
/// cannot reach — `DT_CHECK(0)` in one copy, `llvm_unreachable` in the other — because `arith` has
/// unsigned predicates (`ult`, `ule`, `ugt`, `uge`) that the sentient enum does not.
///
/// ⛔⛔ SO THE UNSIGNED PREDICATES ARE THE REAL CONTENT OF THAT FAILURE ARM, and the way to keep it
/// unreachable is for the island not to express them: [`arith::Op::Compare`] emits `arith.cmpi eq`
/// only — it has no predicate field — so there is nothing here to refuse. If a signed-comparison arm is
/// ever added to that op, this mapping is where the six cases live and an unsigned one must not be
/// added without a sentient counterpart.
#[must_use]
pub const fn predicate(signed: Signed) -> CmpPredicate {
    match signed {
        Signed::Eq => CmpPredicate::Eq,
        Signed::Ne => CmpPredicate::Ne,
        Signed::Lt => CmpPredicate::Slt,
        Signed::Le => CmpPredicate::Sle,
        Signed::Gt => CmpPredicate::Sgt,
        Signed::Ge => CmpPredicate::Sge,
    }
}

/// THE SIX SIGNED COMPARISONS BOTH DIALECTS SHARE.
///
/// ⛔ SIGNED ONLY, AND THE NAME SAYS SO. `arith::CmpIPredicate` also has `ult`/`ule`/`ugt`/`uge`;
/// `sentient`'s `CmpIPredicate` has no unsigned case at all (`SentientTypes.td:474-489`), which is why
/// the reference's mapping has an unreachable failure arm. Naming this type `Signed` makes the omission
/// deliberate rather than forgotten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signed {
    /// `eq`.
    Eq,
    /// `ne`.
    Ne,
    /// `slt`.
    Lt,
    /// `sle`.
    Le,
    /// `sgt`.
    Gt,
    /// `sge`.
    Ge,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 🎯 THE FOUR SCALAR CORRESPONDENCES, AND THE THREE NON-SCALAR ARITH OPS.
    #[test]
    fn the_scalar_correspondences() {
        use crate::islands::dataflow_ir::dialects::Val;
        assert_eq!(
            scalar_for(&arith::Op::Constant {
                result: Val(0),
                value: 7
            }),
            Some(Scalar::Constant)
        );
        assert_eq!(scalar_for(&arith::Op::True { result: Val(0) }), None);
        assert_eq!(
            scalar_for(&arith::Op::Logic {
                result: Val(0),
                kind: arith::LogicKind::And,
                operands: Vec::new()
            }),
            None
        );
    }

    /// 🎯 ONLY `mul` CARRIES ATTRIBUTES ACROSS — the reference's asymmetry, pinned.
    ///
    /// ⛔ IF THIS EVER READS UNIFORM, someone has smoothed the reference and our emission has drifted
    /// from the goldens for a reason that will be hard to place.
    #[test]
    fn only_mul_propagates_attributes() {
        assert!(Scalar::Mul.propagates_attributes());
        assert!(!Scalar::Add.propagates_attributes());
        assert!(!Scalar::Sub.propagates_attributes());
        assert!(!Scalar::Constant.propagates_attributes());
    }

    /// 🎯 THE SIX SIGNED PREDICATES MAP, AND THE MAPPING IS TOTAL.
    ///
    /// ⭐ NO FAILURE ARM, because [`Signed`] admits no unsigned comparison — which is what the
    /// reference's `DT_CHECK(0)` / `llvm_unreachable` was standing in for.
    #[test]
    fn the_six_signed_predicates() {
        assert_eq!(predicate(Signed::Eq), CmpPredicate::Eq);
        assert_eq!(predicate(Signed::Ne), CmpPredicate::Ne);
        assert_eq!(predicate(Signed::Lt), CmpPredicate::Slt);
        assert_eq!(predicate(Signed::Le), CmpPredicate::Sle);
        assert_eq!(predicate(Signed::Gt), CmpPredicate::Sgt);
        assert_eq!(predicate(Signed::Ge), CmpPredicate::Sge);
    }
}
