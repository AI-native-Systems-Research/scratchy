// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/progir/progir.h` — 18 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use super::progir_print::Simplicity;
use crate::generated::DataType;
use crate::islands::progir::ty::OperandValue;
use crate::islands::progir::{Block, BlockKind, Instruction, RegInit, UnitProgram, UnitRegState};

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

impl Instruction {
    /// Replaces: e011_getCommentStr
    ///
    /// The instruction's comment, or the reference's shared `empty` when it has none.
    ///
    /// ⛔ `""` IS ABSENT, NOT PRESENT-AND-BLANK: `hasComment` is `(bool)comment_ && !empty()`
    /// (`progir.h:335`), so the senprog writer suppresses its `  // ` for a comment set to `""`.
    #[must_use]
    pub fn comment_str(&self) -> &str {
        self.comment.as_deref().unwrap_or("")
    }
}

impl RegInit {
    /// Replaces: e015_hasSenDataType
    ///
    /// `senDataType_ != INVALID`.
    ///
    /// ⛔ `INVALID` IS `None` HERE, so the sentinel is gone from the type and the test is `is_some`.
    #[must_use]
    pub const fn has_sen_data_type(&self) -> bool {
        self.sen_data_type.is_some()
    }

    /// Replaces: e016_getSenDataType
    ///
    /// The format this register's content is in, which the reg-init line prints as
    /// `datatype:<name> ` (`dpc.cpp:748-751`) — its only callsite.
    ///
    /// ⛔ THE REFERENCE `DT_ERROR`s WHEN ABSENT; here the absence IS the return, so the prefix is
    /// gated by the very value it prints instead of by a separate predicate.
    #[must_use]
    pub const fn sen_data_type(&self) -> Option<DataType> {
        self.sen_data_type
    }
}

impl UnitProgram {
    /// Replaces: e018_getSimpleInstrVect
    ///
    /// The instructions of a simple CODE graph — `isGraphSimple(true)`, then the
    /// `static_cast<ProgIrCodeBlock *>(head)` that the abort licenses.
    ///
    /// ⛔ TRAP: THE REFERENCE HANDS A **MUTABLE** VECTOR OUT OF A `const` METHOD, and callers assign
    /// through it (`dpc.cpp:221`, `getSimpleInstrVect() = sencomp_instrVec;`). Every reader in the
    /// senprog closure — `dpc.cpp:662`, `progir.cpp:698` and `:726` — only reads, so this is shared.
    #[must_use]
    pub fn simple_instr_vect(&self) -> Option<&[Instruction]> {
        // ⭐ THE VERDICT IS THE GATE AND THE PATTERN IS THE CAST. Any other pair is a graph the
        // reference aborts on, which here is the absence of a vector rather than a stop.
        match (self.simplicity(BlockKind::Code), self.blocks.first()) {
            (Simplicity::Simple, Some(Block::Code(instrs))) => Some(instrs),
            _ => None,
        }
    }

    /// Replaces: e019_getSimpleRegInit
    ///
    /// The same gate one kind over, for a `ProgIrRegGraph` — `static_cast<ProgIrRegBlock *>(head)`.
    ///
    /// ⭐ `Program::reg_state` ALREADY HOLDS THIS FLATTENED, one [`UnitRegState`] per unit, so
    /// `dpc.cpp:730`'s reader takes it from there; this answers for a graph that carries the block.
    #[must_use]
    pub fn simple_reg_init(&self) -> Option<&UnitRegState> {
        match (self.simplicity(BlockKind::RegInit), self.blocks.first()) {
            (Simplicity::Simple, Some(Block::RegInit(state))) => Some(state),
            _ => None,
        }
    }
}

impl Instruction {
    /// Replaces: e020_getTagStr
    ///
    /// The branch label, or `""` — the reference returns a reference to one shared `static empty`
    /// string when `tag_` is null (`progir.h:358`), so a missing tag reads as an empty one.
    #[must_use]
    pub fn tag_str(&self) -> &str {
        self.tag.as_deref().unwrap_or("")
    }

    /// Replaces: e021_hasComment
    ///
    /// `(bool)comment_ && !getCommentStr().empty()` — [`Self::comment_str`] (e011) already reads a
    /// null comment as `""`, so both halves are this one test.
    ///
    /// ⛔ TRAP: A COMMENT IS NEVER LOAD-BEARING — `setComment` drops the text unless
    /// `enable_debug_flag` (`progir.h:323-330`), and senprog trails it as `"  // "` on the
    /// instruction's line when this or `deadCode_` holds (`dpc.cpp:716-717`).
    #[must_use]
    pub fn has_comment(&self) -> bool {
        !self.comment_str().is_empty()
    }

    /// Replaces: e022_hasTag
    ///
    /// ⭐ THIS IS WHAT SENPROG WRITES A LABEL FOR: `if (instr.hasTag()) out << "///" << getTagStr()`
    /// on its own line, before the opcode (`dpc.cpp:663`).
    ///
    /// ⛔ TRAP: AN EMPTY TAG IS NO TAG. `setTag` returns on an empty string (`progir.h:315`), so
    /// `Some(String::new())` is a state only a hand-built instruction reaches — and it answers
    /// `false`, exactly as the reference's `!getTagStr().empty()` does.
    #[must_use]
    pub fn has_tag(&self) -> bool {
        !self.tag_str().is_empty()
    }
}

impl OperandValue {
    /// Replaces: e023_isBool
    ///
    /// `type_ == Type::BOOLEAN` — ⛔ NOT AN INT, though `asBool` delegates to `asInt`
    /// (`progir.h:90-93`): `print` writes a BOOLEAN as `to_string(asBool(id))`, i.e. `0`/`1`
    /// (`progir.cpp:47-51`), and `isInt` still says no.
    #[must_use]
    pub const fn is_bool(&self) -> bool {
        matches!(self, Self::Boolean(_))
    }

    /// Replaces: e024_isFloat
    ///
    /// `type_ == Type::FLOAT` — a 32-bit float, and the only kind `asFloat` admits
    /// (`progir.h:74-77`). ⛔ INT128 is the four-word vector immediate, not a wide float.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::Float(_))
    }
}

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

        assert!(
            descriptive.is_descriptive() && !descriptive.is_tag() && !descriptive.is_variable()
        );
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

    /// e011: a comment is the text, and its absence is the empty string.
    #[test]
    fn a_missing_comment_reads_as_empty() {
        use crate::bridges::progir_to_senprog::test_fixtures::instr;

        let mut with = instr(Vec::new());
        with.comment = Some("spill".to_owned());
        assert_eq!(with.comment_str(), "spill");
        assert_eq!(instr(Vec::new()).comment_str(), "");
    }

    /// e015 and e016: `DataFormats::INVALID` is the absence of a format, and a present one is
    /// returned as it stands.
    #[test]
    fn a_register_format_is_present_or_absent_and_never_a_sentinel() {
        use crate::islands::progir::ty::RegType;
        use crate::islands::sentient::dialects::sentient::RegIndex;

        let reg = |data_type| RegInit {
            file: RegType::Lrf,
            index: RegIndex::at::<0>(),
            value: OperandValue::Int(7),
            sen_data_type: data_type,
        };

        assert!(!reg(None).has_sen_data_type());
        assert_eq!(reg(None).sen_data_type(), None);
        assert!(reg(Some(DataType::Sen169Fp16)).has_sen_data_type());
        assert_eq!(
            reg(Some(DataType::Sen169Fp16)).sen_data_type(),
            Some(DataType::Sen169Fp16)
        );
    }

    /// `getSimpleInstrVect` and `getSimpleRegInit` are one gate two kinds apart: each answers for a
    /// graph of its own kind and, where the reference aborts, for neither.
    #[test]
    fn each_simple_accessor_answers_only_for_its_own_kind_of_graph() {
        use crate::bridges::progir_to_senprog::test_fixtures::instr;

        let one = instr(Vec::new());
        let code = UnitProgram {
            blocks: vec![Block::Code(vec![one.clone()])],
        };
        assert_eq!(code.simple_instr_vect(), Some(&[one][..]));
        assert_eq!(code.simple_reg_init(), None);

        let regs = UnitProgram {
            blocks: vec![Block::RegInit(UnitRegState::new())],
        };
        assert_eq!(regs.simple_reg_init(), Some(&UnitRegState::new()));
        assert_eq!(regs.simple_instr_vect(), None);
    }

    /// ⭐ THE NEGATIVE IS THE EMPTY STRING, NOT THE ABSENCE: `setTag`/`setComment` both refuse an
    /// empty one, so `Some(String::new())` is reachable only by hand and must read as "none".
    #[test]
    fn an_empty_tag_or_comment_is_no_tag_and_no_comment() {
        use crate::bridges::progir_to_senprog::test_fixtures::instr;

        let mut instr = instr(Vec::new());
        assert_eq!(instr.tag_str(), "");
        assert!(!instr.has_tag() && !instr.has_comment());

        instr.tag = Some(String::new());
        instr.comment = Some(String::new());
        assert_eq!(instr.tag_str(), "");
        assert!(!instr.has_tag() && !instr.has_comment());

        instr.tag = Some("L3_loop_end".to_owned());
        instr.comment = Some("row 0 tail".to_owned());
        assert_eq!(instr.tag_str(), "L3_loop_end");
        assert!(instr.has_tag() && instr.has_comment());
    }

    /// A BOOLEAN and a FLOAT are distinct tags, and neither is the INT that `asBool` reads through.
    #[test]
    fn bool_and_float_answer_only_to_their_own_kind() {
        let boolean = OperandValue::Boolean(true);
        let float = OperandValue::Float(1.5);

        assert!(boolean.is_bool() && !boolean.is_float());
        assert!(float.is_float() && !float.is_bool());
        assert!(!OperandValue::Int(1).is_bool() && !OperandValue::Int128([0; 4]).is_float());
    }
}
