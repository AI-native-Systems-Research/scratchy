// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/progir/progir.h` — 18 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use crate::islands::progir::ty::OperandValue;

impl OperandValue {
    /// Replaces: e001_isDescriptive
    ///
    /// `type_ == Type::DESCRIPTIVE` — the enum-like kind, a string underneath. ⭐ THIS IS THE ONE
    /// `convertIr2Senprog` ENCODES RATHER THAN PRINTS: its text is a key into
    /// `typeToFieldEncoding.at(instrType).at(fieldPos)` (`dpc.cpp:691-694`), never the output.
    #[must_use]
    pub const fn is_descriptive(&self) -> bool {
        matches!(self, Self::Descriptive(_))
    }

    /// Replaces: e002_isTag
    ///
    /// `type_ == Type::INSTR_TAG` — a branch label. ⛔ IT RESOLVES TWO WAYS, and which one is the
    /// FIELD's choice, not the tag's: `src0` of a `JCMP`/`JCMPI` goes through `tagToLCCR` and every
    /// other tagged operand through `tagToPC` (`dpc.cpp:695-705`).
    #[must_use]
    pub const fn is_tag(&self) -> bool {
        matches!(self, Self::InstrTag(_))
    }

    /// Replaces: e003_isVariable
    ///
    /// `type_ == Type::VARIABLE` — a name the correction table substitutes later, which senprog
    /// prints between `$` delimiters (`dpc.cpp:687-688`).
    #[must_use]
    pub const fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Replaces: e004_asString
    ///
    /// The text of the three string-typed kinds — `setOperand` admits a string as `DESCRIPTIVE`,
    /// `VARIABLE` or `INSTR_TAG` and nothing else (`progir.h:140-144`). Nothing here is the
    /// reference's `DT_ERROR("OperandAttr: attribute not string")`.
    ///
    /// ⛔ TRAP: THE REFERENCE'S `id` SELECTS AMONG PER-FOLD VALUES — `getValueImpl` reads
    /// `map<SdscFoldId, std::string>` when the attribute `isFolded()` (`progir.h:239-254`). This
    /// island holds ONE value per operand, so there is no id to pass and no folded case to miss.
    #[must_use]
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::Variable(text) | Self::Descriptive(text) | Self::InstrTag(text) => Some(text),
            // ⛔ ENUMERATED, NOT `_`: a tenth kind must break this rather than read as "not a string".
            Self::Float(_)
            | Self::Int(_)
            | Self::Boolean(_)
            | Self::Unknown
            | Self::Int128(_)
            | Self::VariableSymbol(_) => None,
        }
    }
}

// crustify:todo: e011_getCommentStr
//   authority: sys-arch-spec/progir/progir.h:347  (3 lines)  `getCommentStr`

// crustify:todo: e015_hasSenDataType
//   authority: sys-arch-spec/progir/progir.h:119  (1 lines)  `hasSenDataType`

// crustify:todo: e016_getSenDataType
//   authority: sys-arch-spec/progir/progir.h:95  (5 lines)  `getSenDataType`

// crustify:todo: e018_getSimpleInstrVect
//   authority: sys-arch-spec/progir/progir.h:464  (4 lines)  `getSimpleInstrVect`

// crustify:todo: e019_getSimpleRegInit
//   authority: sys-arch-spec/progir/progir.h:497  (4 lines)  `getSimpleRegInit`

// crustify:todo: e020_getTagStr
//   authority: sys-arch-spec/progir/progir.h:345  (1 lines)  `getTagStr`

// crustify:todo: e021_hasComment
//   authority: sys-arch-spec/progir/progir.h:335  (1 lines)  `hasComment`

// crustify:todo: e022_hasTag
//   authority: sys-arch-spec/progir/progir.h:333  (1 lines)  `hasTag`

// crustify:todo: e023_isBool
//   authority: sys-arch-spec/progir/progir.h:108  (1 lines)  `isBool`

// crustify:todo: e024_isFloat
//   authority: sys-arch-spec/progir/progir.h:107  (1 lines)  `isFloat`

// crustify:todo: e025_isInt
//   authority: sys-arch-spec/progir/progir.h:106  (1 lines)  `isInt`

// crustify:todo: e026_isInt128
//   authority: sys-arch-spec/progir/progir.h:109  (1 lines)  `isInt128`

// crustify:todo: e027_isVariableSymbol
//   authority: sys-arch-spec/progir/progir.h:105  (1 lines)  `isVariableSymbol`

// crustify:todo: e029_asInt
//   authority: sys-arch-spec/progir/progir.h:68  (5 lines)  `asInt`

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// The three string kinds are disjoint tags, so each predicate answers for exactly one of them
    /// — `OperandAttr::Type` gives `VARIABLE`, `DESCRIPTIVE` and `INSTR_TAG` distinct values
    /// (`progir.h:46-56`).
    #[test]
    fn each_string_kind_answers_to_exactly_its_own_predicate() {
        let descriptive = OperandValue::Descriptive("be".to_owned());
        let tag = OperandValue::InstrTag("L3_loop_end".to_owned());
        let variable = OperandValue::Variable("weight_base".to_owned());

        assert!(descriptive.is_descriptive() && !descriptive.is_tag() && !descriptive.is_variable());
        assert!(tag.is_tag() && !tag.is_descriptive() && !tag.is_variable());
        assert!(variable.is_variable() && !variable.is_descriptive() && !variable.is_tag());
    }

    /// ⭐ THE NEGATIVE IS THE WHOLE POINT OF `asString`: the reference aborts on a non-string kind,
    /// so a caller reading an `INT` as text is the case that must not produce one.
    #[test]
    fn only_the_string_kinds_have_text() {
        assert_eq!(
            OperandValue::Descriptive("uselccr".to_owned()).as_string(),
            Some("uselccr")
        );
        assert_eq!(
            OperandValue::InstrTag("pt_row0_top".to_owned()).as_string(),
            Some("pt_row0_top")
        );
        assert_eq!(
            OperandValue::Variable("act_addr".to_owned()).as_string(),
            Some("act_addr")
        );
        assert_eq!(OperandValue::Int(7).as_string(), None);
        assert_eq!(OperandValue::Boolean(true).as_string(), None);
        assert_eq!(OperandValue::VariableSymbol(3).as_string(), None);
    }
}
