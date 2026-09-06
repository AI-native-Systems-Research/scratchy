//! PROGIR'S OWN VOCABULARY — the register classes, the operand-field kinds, and what makes a
//! program invalid.
//!
//! Authority: `sys-arch-spec/progir/progir.h` and `sys-arch-spec/arch_enums.h` on the pod.

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
