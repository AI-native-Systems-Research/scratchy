//! PROGIR'S OWN VOCABULARY — the register classes, the operand-field kinds, and what makes a
//! program invalid.
//!
//! Authority: `sys-arch-spec/progir/progir.h` and `sys-arch-spec/arch_enums.h` on the pod.

/// WHICH REGISTER FILE — `RegType` (`sys-arch-spec/arch_enums.h:319-337`).
///
/// ⛔⛔ A **DIFFERENT SET** FROM [`crate::islands::sentient::dialects::sentient::RegType`], AND THE
/// TWO MUST NOT BE CONFLATED. They share eight names (`LRF`, `LAR`, `LBR`, `EAR`, `EBR`, `GTR`,
/// `JCR`, `MVR`) and then diverge in both directions:
///
/// * ProgIR adds `ERAT`, `XRF`, `SPR`, `ARF`, `IRF`, `STATE`, `SCALE` — real files that only exist
///   once registers are real.
/// * Sentient adds `unknown`, `imm`, `lccr`, `xrfrdptr`, `xrfwrptr`, `unrelated` — placeholders and
///   pointer roles that belong to the rung where nothing is assigned yet. `unknown` in particular has
///   no meaning here: a ProgIR register is assigned by definition.
///
/// ⛔⛔ AND `MAX_VALUE = STATE` IS A TRAP IN THE HEADER ITSELF. `STATE` is the fourteenth case and
/// `SCALE` the fifteenth, so `MAX_VALUE` names the *second-to-last* enumerator — any C++ loop written
/// `for (r = LRF; r <= MAX_VALUE; ++r)` silently skips `SCALE`. [`Self::ALL`] is the complete list,
/// and it is the only thing to iterate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegType {
    /// `LRF` — the local register file.
    Lrf,
    /// `LAR`.
    Lar,
    /// `LBR` — ⭐ HOLDS AN INDEX, NOT AN ADDRESS, on the store side.
    Lbr,
    /// `EAR`.
    Ear,
    /// `EBR`.
    Ebr,
    /// `GTR`.
    Gtr,
    /// `JCR` — the jump-condition register.
    Jcr,
    /// `ERAT` — ⛔ NO SENTIENT COUNTERPART.
    Erat,
    /// `MVR` — what `MVLOOPCNT` writes.
    Mvr,
    /// `XRF` — the transposed register file. ⛔ Sentient names its read and write POINTERS, not the
    /// file itself.
    Xrf,
    /// `SPR`.
    Spr,
    /// `ARF` — the accumulator register file.
    Arf,
    /// `IRF`.
    Irf,
    /// `STATE` — ⛔ WHAT THE HEADER'S `MAX_VALUE` POINTS AT, which is not the last case.
    State,
    /// `SCALE` — ⛔ PAST `MAX_VALUE`; see the type's note.
    Scale,
}

impl RegType {
    /// EVERY CASE, in the header's own order.
    ///
    /// ⛔ THE ONLY THING TO ITERATE — `MAX_VALUE` excludes [`Self::Scale`].
    pub const ALL: [RegType; 15] = [
        RegType::Lrf,
        RegType::Lar,
        RegType::Lbr,
        RegType::Ear,
        RegType::Ebr,
        RegType::Gtr,
        RegType::Jcr,
        RegType::Erat,
        RegType::Mvr,
        RegType::Xrf,
        RegType::Spr,
        RegType::Arf,
        RegType::Irf,
        RegType::State,
        RegType::Scale,
    ];

    /// The spelling the printed program uses — `regTypeToString`
    /// (`progir.h:307`, populated in `progir.cpp`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Lrf => "LRF",
            Self::Lar => "LAR",
            Self::Lbr => "LBR",
            Self::Ear => "EAR",
            Self::Ebr => "EBR",
            Self::Gtr => "GTR",
            Self::Jcr => "JCR",
            Self::Erat => "ERAT",
            Self::Mvr => "MVR",
            Self::Xrf => "XRF",
            Self::Spr => "SPR",
            Self::Arf => "ARF",
            Self::Irf => "IRF",
            Self::State => "STATE",
            Self::Scale => "SCALE",
        }
    }
}

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
