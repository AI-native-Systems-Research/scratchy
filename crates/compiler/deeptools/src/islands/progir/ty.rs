//! PROGIR'S OWN VOCABULARY — the register classes, the operand-field kinds, and what makes a
//! program invalid.
//!
//! Authority: `sys-arch-spec/progir/progir.h` and `sys-arch-spec/arch_enums.h` on the pod.

use crate::formats::DataFormat;
use crate::islands::sentient::dialects::sentient::RegType as SenRegType;

/// WHICH REGISTER FILE — ⭐ THE VENDORED ENUM, re-exported.
///
/// ⛔⛔ THIS WAS HAND-TRANSCRIBED HERE AND IS NOW DELETED. `sys-arch-spec` ports `arch_enums.h`'s own
/// `RegType`, and holding a second copy is precisely what that crate exists to prevent — its manifest
/// says so: *"a fact about the machine that two crates each held a copy of is a fact that can disagree
/// with itself"*.
///
/// ⭐ AND IT ALREADY ENCODES THE TRAP I RE-DERIVED. `arch_enums.rs` carries
/// `MAX_VALUE_IS_NOT_THE_MAXIMUM` — the header's `MAX_VALUE = STATE` names the second-to-last
/// enumerator, so a C++ loop bounded by it skips `SCALE`. Two independent readings agreeing is worth
/// more than either, and the vendored one is the copy to keep.
///
/// ⛔ NOT TO BE CONFUSED WITH [`crate::islands::sentient::dialects::sentient::RegType`], which is the
/// SENTIENT dialect's own set — it adds `unknown`/`imm`/`lccr` and the XRF pointers and lacks
/// `ERAT`/`XRF`/`SPR`/`ARF`/`IRF`/`STATE`/`SCALE`. Different rung, different vocabulary.
pub use sys_arch_spec::arch_enums::RegType;

/// ONE OPERAND FIELD'S VALUE — `OperandAttr` (`progir.h:44-57`).
///
/// ⛔⛔ THE DISCRIMINANTS ARE SPARSE AND THE GAPS ARE REAL: `VARIABLE` is 0, then 5, 6, 7, 10, 20,
/// 21, 22, 23 (`progir.h:46-56`). Anything renumbering them contiguously writes a different field
/// kind into every instruction.
///
/// ⭐ A RUST ENUM CARRYING THE VALUE, WHERE THE C++ CARRIES A TAG AND A SEPARATE PAYLOAD. The C++'s
/// accessors are `asString`/`asInt`/`asFloat`/`asInt128`, each of which calls `DT_ERROR` when the tag
/// does not match — six ways to read one field, five of them a runtime abort. Here the tag *is* the
/// payload, so reading the wrong kind is E0308 rather than `OperandAttr: attribute not int`.
#[derive(Debug, Clone, PartialEq)]
pub enum OperandValue {
    /// `VARIABLE` (0) — a name the correction table later substitutes.
    Variable(String),
    /// `FLOAT` (5).
    Float(f32),
    /// `INT` (6).
    Int(i64),
    /// `BOOLEAN` (7).
    Boolean(bool),
    /// `UNKNOWN` (10).
    Unknown,
    /// `DESCRIPTIVE` (20) — enum-like, a string underneath. ⭐ THIS IS WHAT AN ISA FIELD NAME USES.
    Descriptive(String),
    /// `INT128` (21) — four words, for a vector immediate.
    Int128([u32; 4]),
    /// `INSTR_TAG` (22) — a branch target's label, resolved to a PC by `tagToPC`.
    InstrTag(String),
    /// `VARIABLE_SYMBOL` (23) — a symbol id, held as an integer.
    VariableSymbol(i64),
}

impl OperandValue {
    /// The wire tag (`progir.h:46-56`). ⛔ SPARSE — see the type's note.
    #[must_use]
    pub const fn tag(&self) -> u32 {
        match self {
            Self::Variable(_) => 0,
            Self::Float(_) => 5,
            Self::Int(_) => 6,
            Self::Boolean(_) => 7,
            Self::Unknown => 10,
            Self::Descriptive(_) => 20,
            Self::Int128(_) => 21,
            Self::InstrTag(_) => 22,
            Self::VariableSymbol(_) => 23,
        }
    }
}

/// WHY A PROGRAM IS INVALID — `ProgramAndStateInfo::ErrorType` (`progir.h:519-528`).
///
/// ⭐ THE SET IS WORTH HAVING AS A TYPE because it is the reference's own list of what it refuses,
/// and every one of them is a lowering defect on our side rather than a user error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invalid {
    /// `OPCODE_OPERAND` — an operand the opcode does not take.
    OpcodeOperand,
    /// `IBUFF_OVERFLOW` — more than [`super::Program::MAX_INSTRUCTIONS`] on some unit.
    IBuffOverflow,
    /// `LOOP` — a malformed loop.
    Loop,
    /// `PC_TARGET` — a branch to nowhere.
    PcTarget,
    /// `GRAPH` — the block graph is not well formed.
    Graph,
    /// `IMMEDIATE` — an immediate that does not fit its field.
    Immediate,
    /// `REG_INIT` — a register read before anything initialised it.
    RegInit,
}

/// WHICH FOLD OF A FOLDED PROGRAM — `SdscFoldId` (`util/sendefs/sendefs.h:197`).
///
/// ⛔ NOT [`crate::bridges::dataflow_ir_to_sentient::tf_unit_filtering::FoldId`], which bridge 2
/// deliberately narrowed to the single fold its span could reach. An `OperandAttr` genuinely holds
/// one value PER FOLD (`progir.h:266-269`), so this one is the whole index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldId(pub u32);

/// WHETHER AN OPERAND IS ONE VALUE OR ONE VALUE PER FOLD — `OperandAttr`'s `std::variant`
/// (`progir.h:262-269`), whose eight alternatives are four scalars and the same four keyed by fold.
///
/// ⭐ TWO ALTERNATIVES, NOT EIGHT. The variant's scalar/map split is the only distinction that
/// changes what gets emitted; which of the four payload types it is, is already [`OperandValue`].
#[derive(Debug, Clone, PartialEq)]
pub enum PerFold {
    /// One value, on every fold.
    Every(OperandValue),
    /// One value per fold, in fold order.
    ByFold(Vec<(FoldId, OperandValue)>),
}

/// ONE OPERAND — `OperandAttr` whole (`progir.h:43-270`): the value, and the format it is in.
#[derive(Debug, Clone, PartialEq)]
pub struct Operand {
    /// The value, common or per-fold.
    pub value: PerFold,
    /// `senDataType_` — ⛔ `None` IS `DataFormats::INVALID`, which is what `hasSenDataType` tests
    /// for (`progir.h:119`), so the absence has no spelling of its own.
    pub format: Option<DataFormat>,
}

impl Default for Operand {
    /// `OperandAttr()` — ⛔ `type_ = UNKNOWN` AND NO FORMAT (`progir.h:257-271`), not a zero.
    fn default() -> Self {
        Operand {
            value: PerFold::Every(OperandValue::Unknown),
            format: None,
        }
    }
}

impl Operand {
    /// ONE VALUE ON EVERY FOLD — `OperandAttr(input)` with no fold id.
    #[must_use]
    pub const fn every(value: OperandValue) -> Operand {
        Operand {
            value: PerFold::Every(value),
            format: None,
        }
    }

    /// `asString` (`progir.h:62-66`) — the string a VARIABLE, DESCRIPTIVE or INSTR_TAG operand holds.
    ///
    /// ⛔ `None` WHERE THE REFERENCE ABORTS: any other type is *"attribute not string"*, and a folded
    /// operand asked without a fold id is *"folded but no SdscFoldId to access"* (`progir.h:247`).
    #[must_use]
    pub fn as_string(&self, id: Option<FoldId>) -> Option<&str> {
        let value = match (&self.value, id) {
            (PerFold::Every(value), _) => value,
            (PerFold::ByFold(folds), Some(id)) => &folds.iter().find(|(at, _)| *at == id)?.1,
            (PerFold::ByFold(_), None) => return None,
        };
        match value {
            OperandValue::Variable(text)
            | OperandValue::Descriptive(text)
            | OperandValue::InstrTag(text) => Some(text),
            _ => None,
        }
    }

    /// `setSenDataType` (`progir.h:180`).
    #[must_use]
    pub fn in_format(self, format: DataFormat) -> Operand {
        Operand {
            value: self.value,
            format: Some(format),
        }
    }

    /// `setOperand(input, id)` — `setOperandImpl` (`progir.h:212-234`).
    ///
    /// ⛔⛔ THE TWO DIRECTIONS ARE NOT SYMMETRIC. With a fold id it MERGES into the per-fold map, and
    /// a fold id arriving at a common value DISCARDS that value; without one it REPLACES everything,
    /// so a common write after per-fold writes throws the whole map away.
    pub fn set(&mut self, id: Option<FoldId>, value: OperandValue) {
        match id {
            None => self.value = PerFold::Every(value),
            Some(id) => match &mut self.value {
                PerFold::ByFold(folds) => match folds.iter_mut().find(|(at, _)| *at == id) {
                    Some(entry) => entry.1 = value,
                    None => folds.push((id, value)),
                },
                PerFold::Every(_) => self.value = PerFold::ByFold(vec![(id, value)]),
            },
        }
    }
}

/// WHICH REGISTER FILE A SENTIENT LOCALE NAMES — `ProgramAndStateInfo::stringToRegType`
/// (`progir.cpp:670-675`), which is `regTypeToString` flipped, applied to the uppercased locale.
///
/// ⛔ `None` IS WHERE THE REFERENCE ABORTS: `.at()` throws for a locale with no row — `unknown` and
/// `unrelated` (unassigned), `imm` (an instruction field, not a file), `lccr`, and the two XRF
/// POINTERS, which are not the `XRF` the table names.
///
/// ⛔⛔ AND `SCALE` HAS NO ROW EITHER — the table stops at `STATE`, the same off-by-one
/// [`sys_arch_spec::arch_enums::MAX_VALUE_IS_NOT_THE_MAXIMUM`] records, leaking into the strings.
#[must_use]
pub const fn reg_file_of(locale: SenRegType) -> Option<RegType> {
    match locale {
        SenRegType::Lrf => Some(RegType::Lrf),
        SenRegType::Lar => Some(RegType::Lar),
        SenRegType::Lbr => Some(RegType::Lbr),
        SenRegType::Ear => Some(RegType::Ear),
        SenRegType::Ebr => Some(RegType::Ebr),
        SenRegType::Gtr => Some(RegType::Gtr),
        SenRegType::Mvr => Some(RegType::Mvr),
        SenRegType::Jcr => Some(RegType::Jcr),
        SenRegType::Unknown
        | SenRegType::Imm
        | SenRegType::Lccr
        | SenRegType::XrfRdPtr
        | SenRegType::XrfWrPtr
        | SenRegType::Unrelated => None,
    }
}
