//! THE PROGIR ISLAND — machine instructions with their registers ASSIGNED, and the program state
//! that rides with them.
//!
//! ```text
//! SentientIR ──D76 SentientToProgIR──► ProgIR ──dip-generate-init-packet──► the packet
//!                                      (here)
//! ```
//!
//! Authority: `sys-arch-spec/progir/progir.h` on the pod.
//!
//! ⛔⛔ **PROGIR IS NOT AN MLIR DIALECT**, and that is the structural fact about this rung.
//! `dcc/src/Dialect/` holds Agen, Dataflow, Sentient, Trace and Uniform — there is no ProgIR among
//! them. `SentientToProgIR` fills in a plain C++ structure, `std::map<int, ProgramAndStateInfo>` keyed
//! by core id, and the MLIR module it leaves behind holds only [`dialects::init`]'s reference to it.
//! So the ladder leaves MLIR here, which is exactly why our Rust could already meet it at the bottom.
//!
//! ⭐⭐ AND THE WHOLE THING IS TINY. `kMaxCompIBuff = 256` — *"Maximum number of instructions on any
//! unit"* — with `kMaxCompRegs = 128` (`progir.h:506-510`). A complete int8 batched matmul compiles to
//! **eight** instructions on one unit. Whatever the rungs above cost, this is the size of what they
//! produce.

pub mod dialects;
pub mod print;
pub mod ty;

use crate::arch::Arch;
use crate::generated::DataType;
use crate::islands::sentient::dialects::sentient::RegIndex;
use crate::model::Model;
use crate::workload::Workload;
use sys_arch_spec::progir::{MAX_INSTRUCTIONS_PER_UNIT, MAX_REGISTERS_PER_UNIT};
use sys_arch_spec::regfile::{Component, max_ibuff_entries};
use ty::{Invalid, OperandValue, RegType};

/// ONE REGISTER'S INITIAL CONTENT — one entry of a unit's register state.
///
/// ⭐ A NAMED STRUCT BECAUSE THE FILE AND THE INDEX ARE TWO DIFFERENT THINGS. `addRegInit` takes
/// `(unsigned regNum, const OperandAttr &regContent, RegType regType = RegType::LRF)`
/// (`progir.h:494-500`) — note the DEFAULTED file, which means a caller passing only a number
/// silently initialises an LRF. Naming both here removes the default.
#[derive(Debug, Clone, PartialEq)]
pub struct RegInit {
    /// Which file.
    pub file: RegType,
    /// Which register within it.
    ///
    /// ⛔⛔ BOUNDED BY THE TYPE, NOT BY A COMMENT. `kMaxCompRegs` is the width of the
    /// `std::bitset<kMaxCompRegs>` that records which registers are defined (`progir.h:302-304`), so
    /// an index past it cannot even be *recorded* as initialised — it would silently fall outside the
    /// bitset. [`RegIndex`] is the Sentient rung's type, reused here because it is the same field with
    /// the same cap, one rung further down.
    pub index: RegIndex,
    /// What it starts as.
    pub value: OperandValue,
    /// `senDataType_` — the format the content is in (`progir.h:271`), which the senprog reg-init
    /// line prints as a `datatype:<name> ` prefix when it is set (`dpc.cpp:748-751`).
    ///
    /// ⛔ `None` IS THE REFERENCE'S `DataFormats::INVALID` DEFAULT rather than a missing field: that
    /// sentinel is precisely what `hasSenDataType` tests (`progir.h:119`), so an `Option` here
    /// deletes the sentinel instead of carrying it.
    pub sen_data_type: Option<DataType>,
}

/// WHAT ONE UNIT'S REGISTERS START AS — the reference's own `UnitRegState`
/// (`typedef std::map<RegType, std::map<unsigned int, OperandAttr>>`, `progir.h:361`).
///
/// ⭐ THE REFERENCE NAMES THIS TYPE, so naming it here is faithful rather than tidying: it is what
/// `getSimpleRegInit` returns and what `RegStateInfo` maps a unit to.
pub type UnitRegState = Vec<RegInit>;

/// ONE INSTRUCTION'S OPCODE — `InstrInfo::instn_`, an `OpCodeT` (`progir.h:280`).
///
/// ⛔⛔ THIS WAS `OpCode(pub u16)` — AN UNBOUNDED PUBLIC FIELD admitting any number as an opcode. It is
/// now the vendored enum's 91 variants, so an opcode no unit has cannot be written down. The `TODO`
/// that stood here was wrong about the work: the table was not owed, it was already ported.
///
/// ⭐ THE ENUM IS THE AUTHORITY FOR WHAT IS AN OPCODE, and `deeptools`' own `build.rs` already relies
/// on that — it emits `InstOpCode::FMA` for the mnemonics the `.ddl`/`.smc` templates name, so a
/// template naming a non-opcode fails to compile.
pub use sys_arch_spec::InstOpCode as OpCode;

/// WHICH FIELD OF AN INSTRUCTION — the key of `InstrInfo::instFields_` (`progir.h:283`), an
/// `OperandT`.
///
/// ⛔⛔ ALSO WAS AN UNBOUNDED `pub u16`. Now the vendored 89-operand enum — the one whose two copies
/// drifting is the reason `sys-arch-spec` is a crate at all.
pub use sys_arch_spec::operand::Operand as OperandField;

/// ONE INSTRUCTION — `InstrInfo` (`progir.h:274-357`).
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// `instn_`.
    pub opcode: OpCode,
    /// `symbolicOpCode` — ⭐ THE OPCODE ITSELF CAN BE A SYMBOL the correction table fills in, not
    /// only its operands (`progir.h:281`).
    pub symbolic_opcode: Option<i64>,
    /// `instFields_` — the operand fields, in field order.
    pub fields: Vec<(OperandField, OperandValue)>,
    /// `deadCode_`.
    pub dead: bool,
    /// `tag_` — a branch label, which `tagToPC` resolves (`progir.h:290`).
    pub tag: Option<String>,
    /// `comment_` — ⛔ DEBUG-ONLY IN THE REFERENCE: `setComment` drops the text unless
    /// `enable_debug_flag` (`progir.h:322-330`), so a comment is never load-bearing.
    pub comment: Option<String>,
}

/// ONE BLOCK OF A UNIT'S PROGRAM — `ProgIrBlock` and its five subclasses
/// (`progir.h:364-405`).
///
/// ⛔⛔ **NESTED, NOT A `prev`/`next` GRAPH**, AND THAT IS A DELIBERATE DEPARTURE. The reference holds
/// `std::vector<ProgIrBlock *> next, prev` plus `ifParents`, `forParents` and `pendingCondEnds`
/// (`progir.h:407-414`) because it BUILDS the graph incrementally at runtime —
/// `startForLoop`/`closeForLoop`/`startIf`/`addElseIf`/`closeIfElse` are the incremental API, and the
/// parent stacks exist to remember where an unfinished region opened.
///
/// ⭐ WE HOLD THE FINISHED NESTING, so none of that bookkeeping has a job: a closed loop is a `body`,
/// and `pendingCondEnds` — "condition_end blocks that still don't have a next" — describes a state
/// that cannot exist here. This is the same reason the DataflowIR island has no `OpBuilder`.
///
/// ⛔ `CONDITION_END` AND `FORLOOP_END` ARE NOT VARIANTS HERE for the same reason: they are the
/// reference's markers for where a region closes, which nesting states structurally. `DUMMY` is the
/// base class's default and never a real block.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// `CODE` — a run of instructions (`ProgIrCodeBlock`, `progir.h:392`).
    Code(Vec<Instruction>),
    /// `FORLOOP` — a counted loop and what it encloses (`ProgIrForLoopBlock`, `progir.h:386`).
    ForLoop {
        /// The loop iterator's name, as `startForLoop` takes it.
        iterator: String,
        /// Its start.
        start: OperandValue,
        /// Its end.
        end: OperandValue,
        /// What it encloses.
        body: Vec<Block>,
    },
    /// `CONDITION` — an if/else-if/else chain (`ProgIrConditionBlock`, `progir.h:381`).
    ///
    /// ⭐ A CHAIN, NOT A PAIR: `addElseIf` may be called repeatedly and `addElse` is
    /// `addElseIf("")` — an else is an else-if with an empty condition (`progir.h:433`).
    Condition {
        /// One arm per condition, in order; an empty expression is the `else`.
        arms: Vec<(String, Vec<Block>)>,
    },
    /// `REGINIT` — what must be in a register before anything reads it (`ProgIrRegBlock`,
    /// `progir.h:397`).
    RegInit(UnitRegState),
    /// `VARDEF` — the variable definitions (`ProgIrVarBlock`, `progir.h:402`).
    VarDef(Vec<(String, String)>),
}

/// WHICH KIND OF BLOCK — `ProgIrBlock::Type` (`progir.h:365-374`), which is also what a graph fixes
/// as its own `graphType` to say which kind its single block must be (`progir.h:418`).
///
/// ⛔ FIVE, NOT EIGHT: `CONDITION_END`, `FORLOOP_END` and `DUMMY` are not [`Block`] variants — see
/// there for why nesting states what the reference's end markers and default do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// `CODE`.
    Code,
    /// `FORLOOP`.
    ForLoop,
    /// `CONDITION`.
    Condition,
    /// `REGINIT`.
    RegInit,
    /// `VARDEF`.
    VarDef,
}

impl Block {
    /// WHICH KIND THIS IS — the reference's `type` field, which its subclass constructor fixes.
    #[must_use]
    pub const fn kind(&self) -> BlockKind {
        match self {
            Block::Code(_) => BlockKind::Code,
            Block::ForLoop { .. } => BlockKind::ForLoop,
            Block::Condition { .. } => BlockKind::Condition,
            Block::RegInit(_) => BlockKind::RegInit,
            Block::VarDef(_) => BlockKind::VarDef,
        }
    }

    /// HOW MANY INSTRUCTIONS THIS BLOCK AND EVERYTHING IT ENCLOSES HOLD.
    ///
    /// ⛔ COUNTS THE LOOP BODY **ONCE**, NOT ONCE PER TRIP. The instruction buffer holds the program,
    /// not its execution, so a loop of two instructions running a thousand times is two.
    #[must_use]
    pub fn instructions(&self) -> usize {
        match self {
            Block::Code(instrs) => instrs.len(),
            Block::ForLoop { body, .. } => body.iter().map(Block::instructions).sum(),
            Block::Condition { arms } => arms
                .iter()
                .map(|(_, blocks)| blocks.iter().map(Block::instructions).sum::<usize>())
                .sum(),
            // Neither states an instruction: a register initialisation is program STATE the packet
            // carries, and a variable definition is a name.
            Block::RegInit(_) | Block::VarDef(_) => 0,
        }
    }
}

/// ONE UNIT'S PROGRAM — a `ProgIrCodeGraph` (`progir.h:455-468`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnitProgram {
    /// The blocks, in order.
    pub blocks: Vec<Block>,
}

impl UnitProgram {
    /// HOW MANY INSTRUCTIONS THIS UNIT HOLDS.
    #[must_use]
    pub fn instructions(&self) -> usize {
        self.blocks.iter().map(Block::instructions).sum()
    }
}

/// A UNIT WHOSE PROGRAM DOES NOT FIT ITS INSTRUCTION BUFFER.
///
/// ⛔ A NAMED STRUCT, NOT A TRIPLE. `(Component, usize, u16)` puts a count and a bound side by side as
/// two bare integers — the transposition this crate's newtype rule exists to prevent, and the reader
/// has nothing but position to tell which is which.
///
/// ⭐ IT CARRIES THE BOUND IT BROKE, because "the PT program is 200 instructions" is not actionable
/// without "and the PT holds 128" — and the bound differs per component and per arch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow {
    /// Which unit.
    pub unit: Component,
    /// How many instructions its program holds.
    pub instructions: usize,
    /// How many that unit's buffer takes — `max_ibuff_entries(unit)`.
    pub bound: u16,
}

/// ONE PROGRAM — `ProgramAndStateInfo` (`progir.h:505-556`).
///
/// # 🛑 THE CONST-GENERIC TRAITS RIDE ALL THE WAY DOWN
///
/// ⭐ THE SAME THREE AS THE RUNGS ABOVE, and they still guard something here: [`Self::MAX_REGISTERS`]
/// bounds what the ISA can name, but what a given unit HAS is an arch fact, so a program allocated
/// against one arch's register-file depths must not be readable as another's.
///
/// ⚠️ **TODO(quant):** a fourth `Q: Quant`, as on [`crate::islands::sentient::Program`].
#[derive(Debug, Clone, PartialEq)]
pub struct Program<A: Arch, M: Model, W: Workload> {
    /// One program per unit — `senCompProgram_`, whose key is a `SenComponents`
    /// (`progir.h:512`).
    ///
    /// ⛔ KEYED BY UNIT, AND THE HALVES STAY SEPARATE. An `l3lu` and an `l3su` are two entries;
    /// merging a load unit's and a store unit's instructions into one program is exactly the mistake
    /// `Component` keeping `Lxlu`/`Lxsu`/`L0lu`/`L0su` apart prevents one rung up.
    ///
    /// ⛔⛔ AND THE KEY IS AN **EXECUTING** UNIT, NOT ANY UNIT. This was `DfirUnit`, which also names
    /// `Lx` and `Hbm` — memories, which hold no instructions and have no instruction buffer. So a
    /// program could be filed against a memory, and no bound could be looked up for it.
    /// [`Component`] is the nine units that execute, which makes [`max_ibuff_entries`] total.
    pub per_unit: Vec<(Component, UnitProgram)>,
    /// `regState_` — a `RegStateInfo`, which is `map<SenComponents, ProgIrRegGraph>`
    /// (`progir.h:502`): per unit, what its registers start as.
    pub reg_state: Vec<(Component, UnitRegState)>,
    /// `variableDefinitions_` — a `ProgIrVarGraph` (`progir.h:306`).
    pub variable_definitions: Vec<(String, String)>,
    /// The arch, model and rung this was compiled for.
    pub bound: core::marker::PhantomData<(A, M, W)>,
}

impl<A: Arch, M: Model, W: Workload> Program<A, M, W> {
    /// THE WORST CASE ACROSS EVERY UNIT — `kMaxCompIBuff` (`progir.h:506-507`), whose own comment is
    /// *"Maximum number of instructions on any unit"*.
    ///
    /// ⛔⛔ **A MAXIMUM OVER UNITS IS NOT ANY UNIT'S LIMIT**, AND USING IT AS ONE WAS A REAL DEFECT.
    /// This constant was the only bound [`Self::overflowing`] applied, so a 200-instruction PT program
    /// passed — while `max_ibuff_entries(Component::Pt)` is **128**. `sysdef.cpp:451-500` gives 256 to
    /// the L3 halves, 128 to PT/PE/SFP/L0/LXSU, and to LXLU 256 on SEN1P5 but 128 otherwise. So the
    /// real bound is per component AND per arch, and only the L3 halves ever reach this number.
    ///
    /// ⭐ KEPT BECAUSE IT IS THE `std::bitset` WIDTH THE REFERENCE ALLOCATES, which is a real fact about
    /// the array — just not about a unit.
    ///
    /// ⛔⛔ AND IT IS RE-EXPORTED, NOT RESTATED. `sys_arch_spec::progir` holds this number for exactly
    /// this reason, and its own doc says the bound *"was written out twice, verbatim — 256 and 128 in
    /// the compiler's island 4 and again in the model's `prog_ir_graph`"*, whose symptom is a compiler
    /// emitting programs the model silently truncates. Writing `256` here made it a third copy, inside
    /// the crate that exists to prevent the second.
    pub const WORST_CASE_INSTRUCTIONS: usize = MAX_INSTRUCTIONS_PER_UNIT;

    /// HOW MANY REGISTERS ANY UNIT MAY USE — `kMaxCompRegs` (`progir.h:508-509`).
    ///
    /// ⛔ THE WIDTH OF A `std::bitset<kMaxCompRegs>` IN `RegDefs` (`progir.h:302-304`), so it is a
    /// hard bound on the allocator rather than a suggestion.
    ///
    /// ⛔ RE-EXPORTED, NOT RESTATED — see [`Self::WORST_CASE_INSTRUCTIONS`].
    pub const MAX_REGISTERS: usize = MAX_REGISTERS_PER_UNIT;

    /// EVERY UNIT WHOSE PROGRAM OVERFLOWS THE INSTRUCTION BUFFER.
    ///
    /// ⭐⭐ THIS IS THE ONE LAW OF THIS ISLAND THAT CAN ACTUALLY FAIL, and it is the reference's own
    /// `IBUFF_OVERFLOW` (`progir.h:522`). Unlike the rung below — where conservation is
    /// *unrepresentable* and so its validator is vacuous — a program that is too long is perfectly
    /// expressible and has to be checked.
    ///
    /// ⛔ RETURNS THE OFFENDERS, NOT A BOOL. "Some unit overflowed" is not a diagnosis anyone can act
    /// on; which unit, and by how much, is.
    #[must_use]
    pub fn overflowing(&self) -> Vec<Overflow> {
        self.per_unit
            .iter()
            .filter_map(|(unit, program)| {
                // ⛔ THE UNIT'S OWN BOUND, NOT THE WORST CASE. See [`Self::WORST_CASE_INSTRUCTIONS`].
                let bound = max_ibuff_entries(*unit);
                let instructions = program.instructions();
                (instructions > bound as usize).then_some(Overflow {
                    unit: *unit,
                    instructions,
                    bound,
                })
            })
            .collect()
    }

    /// WHAT THE REFERENCE WOULD REFUSE THIS PROGRAM FOR, as far as this island can tell.
    ///
    /// ⛔⛔ **THIS IS NOT `checkProgramValidity`**, AND SAYING SO MATTERS. The reference's check
    /// (`progir.h:529-531`) takes `isa_per_unit` and the fold ids and can therefore test
    /// `OPCODE_OPERAND`, `PC_TARGET`, `IMMEDIATE` and `REG_INIT` — every one of which needs the ISA
    /// tables this crate does not have yet. What is checkable here is size, and a green result from
    /// this function means only that.
    ///
    /// ⛔ A PARTIAL CHECK REPORTED AS A FULL ONE IS THE FAILURE MODE THIS CRATE HAS ALREADY PAID FOR
    /// TWICE, so the return names the single law it tested rather than "valid".
    pub fn size_verdict(&self) -> Result<(), (Invalid, Vec<Overflow>)> {
        let over = self.overflowing();
        if over.is_empty() {
            Ok(())
        } else {
            Err((Invalid::IBuffOverflow, over))
        }
    }
}
