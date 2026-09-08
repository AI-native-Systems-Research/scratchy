//! THE SCALAR INSTRUCTIONS — the jumps, the compares, the adds and subs, and the sync.
//!
//! ⛔ THE LDST FAMILY IS ONE OPCODE AND THE COMPONENT DECIDES ITS ROLE (`isa.cpp:952`): the
//! field names `ldtype`/`consumertag` are picked BY COMPONENT, not by opcode. A misport here
//! writes the wrong field name into every load/store.
//! ⛔ THE LBR HOLDS AN INDEX, NOT AN ADDRESS — measured on an EBR-matched op-11 diff.
//!
//! 16 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e001_ConstructNOPInstr` | 0 | 10 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:329` |
//! | `e002_ConstructReturnInstr` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:340` |
//! | `e042_ConstructMVLoopInstr` | 1 | 63 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:48` |
//! | `e043_ConstructAssignInstr` | 1 | 204 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:113` |
//! | `e044_ConstructBranchExitInstr` | 1 | 9 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:319` |
//! | `e045_ConstructSyncInstr` | 1 | 96 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:348` |
//! | `e046_ConstructJMPInstr` | 1 | 11 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:446` |
//! | `e047_ConstructJCMPInstr` | 1 | 227 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:458` |
//! | `e048_ConstructJADDInstr` | 1 | 33 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:687` |
//! | `e049_ConstructXRFADDInstr` | 1 | 47 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:892` |
//! | `e050_ConstructXRFADDInstr` | 1 | 22 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:940` |
//! | `e051_ConstructJSUBInstr` | 1 | 34 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:964` |
//! | `e081_ConstructLRFADDInstr` | 2 | 88 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:721` |
//! | `e082_ConstructLARorEARADDInstr` | 2 | 81 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:810` |
//! | `e083_ConstructLRFSUBInstr` | 2 | 136 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:999` |
//! | `e084_ConstructLARorEARSUBInstr` | 2 | 46 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1136` |

use crate::arch::{Arch, IsaGen};
use crate::bridges::sentient_to_progir::construct::reg_init::{
    ImmSource, add_to_regs_to_init, fill_imm_field,
};
use crate::bridges::sentient_to_progir::construct::{
    boolean, descriptive, instr_tag, int, variable_symbol,
};
use crate::bridges::sentient_to_progir::lower::labels_and_regs::address_scale;
use crate::bridges::sentient_to_progir::state::{
    CopyOps, LabelCounter, Labels, OpSite, RegsToInit, UnitKey,
};
use crate::bridges::sentient_to_progir::uniform::instr::{
    FoldConstant, MapMode, MappedEntry, OperandMapRefusal, UniformInstrInfo,
    add_entry_to_operand_map,
};
use crate::bridges::sentient_to_progir::utils::{AddrSpace, addr_wraparounded};
use crate::formats::Bits;
use crate::islands::progir::ty::Operand;
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{
    CmpPredicate, Consumer, Reg, RegIndex, RegType as SenRegType, SyncMode,
};
use crate::units::Core;
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e001_ConstructNOPInstr
///
/// The do-nothing instruction, carrying whatever the caller wants said about why it is there.
///
/// ⛔ AN EMPTY `comment_str` MEANS ABSENT, NOT AN EMPTY COMMENT — it selects the literal `"NOP"`
/// (`ConstructProgIRHelper.cpp:333-336`), so `Some("")` is the same call as `None` here too.
///
/// ⭐ THE REFERENCE'S `comp` IS UNUSED: a NOP is the same instruction on every unit.
#[must_use]
pub fn construct_nop_instr(comment: Option<&str>) -> UniformInstrInfo {
    let comment = comment.filter(|text| !text.is_empty());
    UniformInstrInfo::of(OpCode::NOP).with_common_comment(comment.unwrap_or("NOP"))
}

/// Replaces: e002_ConstructReturnInstr
///
/// The instruction every unit's program ends on. ⭐ `comp` is unused in the reference too.
#[must_use]
pub fn construct_return_instr() -> UniformInstrInfo {
    UniformInstrInfo::of(OpCode::RETURN).with_common_comment("end of the program")
}

/// `is_any_of(comp, LXLU, LXSU) ? "lrfimm" : "imm"` — the LX halves give the LRF's immediate a field
/// of its own (`ConstructProgIRHelper.cpp:741`).
const fn lrf_imm_field(comp: Component) -> OperandField {
    match comp {
        Component::Lxlu | Component::Lxsu => OperandField::Lrfimm,
        _ => OperandField::Imm,
    }
}

/// AN ADD OR SUB'S INSTRUCTIONS, THE COPIES THEY COST AND WHAT A PER-UNIT MAP COULD NOT HOLD.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarUpdate {
    /// ⛔ THE COPY COMES FIRST when there is one: the modify writes the register it reads.
    pub instrs: Vec<UniformInstrInfo>,
    /// `++num_copy_ops_` — counted for every caller, where the reference counts only under
    /// `cq_stats`, which is one of the three things that also turn `full_reg_init` on.
    pub copy_ops: CopyOps,
    /// The offenders, in entry order.
    pub refused: Vec<OperandMapRefusal>,
}

impl ScalarUpdate {
    /// The instructions, and the copy that had to precede them.
    fn of(
        copy: Option<UniformInstrInfo>,
        instrs: Vec<UniformInstrInfo>,
        refused: Vec<OperandMapRefusal>,
    ) -> ScalarUpdate {
        ScalarUpdate {
            copy_ops: CopyOps(u32::from(copy.is_some())),
            instrs: copy.into_iter().chain(instrs).collect(),
            refused,
        }
    }
}

/// The `*REGCOPY` that puts a source in the target register first — ⛔ THE TARGET IS `src0` (`:751-752`).
fn reg_copy(
    opcode: OpCode,
    comment: &str,
    tgt: Option<RegIndex>,
    src: Option<RegIndex>,
) -> UniformInstrInfo {
    let mut copy = UniformInstrInfo::of(opcode).with_common_comment(comment);
    copy.set_common_field(OperandField::Src0, index_field(tgt));
    copy.set_common_field(OperandField::Src1, index_field(src));
    copy
}

/// `LRFREGCOPY` (`:750-756`).
fn lrf_copy(tgt: Option<RegIndex>, src: Option<RegIndex>) -> UniformInstrInfo {
    reg_copy(OpCode::LRFREGCOPY, "LRF <- LRF", tgt, src)
}

/// The disagreeing units of a constant map, as offenders.
fn folding_needed(units: Vec<UnitKey>) -> Vec<OperandMapRefusal> {
    units
        .into_iter()
        .map(OperandMapRefusal::FoldingNeeded)
        .collect()
}

impl AddrFile {
    /// `getRegLocale()` — what `stringifySentientRegType` spells into the comment (`:813`).
    const fn locale(self) -> SenRegType {
        match self {
            AddrFile::Lar => SenRegType::Lar,
            AddrFile::Ear => SenRegType::Ear,
        }
    }

    /// `is_ear ? EARREGCOPY : LARREGCOPY`, with its comment (`:842-848`).
    const fn copy(self) -> (OpCode, &'static str) {
        match self {
            AddrFile::Lar => (OpCode::LARREGCOPY, "LAR <- LAR"),
            AddrFile::Ear => (OpCode::EARREGCOPY, "EAR <- EAR"),
        }
    }
}

/// `src1` ON A `MOD*REG` — the target must be one of its own operands, so the OTHER operand goes in
/// `src1` and a third register is reached by copying into the target first (`:786-800`).
fn mod_reg_src1(
    tgt: Option<RegIndex>,
    src0: Option<RegIndex>,
    src1: Option<RegIndex>,
    copy: &mut Option<UniformInstrInfo>,
    copy_of: (OpCode, &'static str),
) -> Option<RegIndex> {
    if src0 == tgt {
        src1
    } else if src1 == tgt {
        src0
    } else {
        *copy = Some(reg_copy(copy_of.0, copy_of.1, tgt, src0));
        src1
    }
}

/// WHAT AN LRF ADD ADDS — the three locale pairs of `:743-801`, its `else` an `emitError`.
#[derive(Debug, Clone, PartialEq)]
pub enum LrfAddOperands {
    /// `lrf + imm`.
    RegImm {
        /// `getValueRegIndex(inp1)`.
        src0: Option<RegIndex>,
        /// `inp2`'s defining op.
        imm: ImmSource,
    },
    /// `imm + lrf`.
    ImmReg {
        /// `inp1`'s defining op.
        imm: ImmSource,
        /// `getValueRegIndex(inp2)`.
        src1: Option<RegIndex>,
    },
    /// `lrf + lrf`.
    RegReg {
        /// `getValueRegIndex(inp1)`.
        src0: Option<RegIndex>,
        /// `getValueRegIndex(inp2)`.
        src1: Option<RegIndex>,
    },
}

/// AN LRF ADD — ⛔ `tgt` IS A REGISTER THE INSTRUCTION ITSELF NAMES, not a separate destination.
#[derive(Debug, Clone, PartialEq)]
pub struct LrfAdd {
    /// `getValueRegIndex(getResult())`.
    pub tgt: Option<RegIndex>,
    /// The two operands.
    pub operands: LrfAddOperands,
}

/// Replaces: e081_ConstructLRFADDInstr
///
/// The add that steps an LRF pointer, and the copy that first puts a source in the target register.
///
/// ⛔ `MODLRF*`'s TARGET MUST BE ONE OF ITS OWN OPERAND REGISTERS (`:748`), so a third register costs
/// a `LRFREGCOPY` in front.
/// ⚠️ AND THE `imm + lrf` ARM PUTS THE OPERAND'S REGISTER IN `src0`, NOT THE TARGET'S (`:780`) — with
/// a copy in front it then modifies the source register and leaves the target holding the copy.
#[must_use]
pub fn construct_lrf_add_instr<A: Arch>(
    add: &LrfAdd,
    comp: Component,
    element_size: Bits,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> ScalarUpdate {
    let lrf = |index| Reg {
        locale: SenRegType::Lrf,
        index,
    };
    if full_reg_init {
        let mut regs = match &add.operands {
            LrfAddOperands::RegImm { src0, .. } => vec![lrf(*src0)],
            LrfAddOperands::ImmReg { src1, .. } => vec![lrf(*src1)],
            LrfAddOperands::RegReg { src0, src1 } => vec![lrf(*src0), lrf(*src1)],
        };
        regs.push(lrf(add.tgt));
        add_to_regs_to_init(units, &regs, regs_to_init);
    }
    let scale = address_scale::<A>(comp);
    let imm_field = lrf_imm_field(comp);
    let mut refused = Vec::new();
    let mut copy = None;
    let instr = match &add.operands {
        LrfAddOperands::RegImm { src0, imm } => {
            let mut instr = UniformInstrInfo::of(OpCode::MODLRFIMM).with_common_comment("lrf add");
            instr.set_common_field(OperandField::Src0, index_field(add.tgt));
            refused = fill_imm_field::<A>(&mut instr, imm_field, imm, comp, element_size, scale);
            if *src0 != add.tgt {
                copy = Some(lrf_copy(add.tgt, *src0));
            }
            instr
        }
        LrfAddOperands::ImmReg { imm, src1 } => {
            let mut instr = UniformInstrInfo::of(OpCode::MODLRFIMM).with_common_comment("lrf add");
            instr.set_common_field(OperandField::Src0, index_field(*src1));
            refused = fill_imm_field::<A>(&mut instr, imm_field, imm, comp, element_size, scale);
            if *src1 != add.tgt {
                copy = Some(lrf_copy(add.tgt, *src1));
            }
            instr
        }
        LrfAddOperands::RegReg { src0, src1 } => {
            let mut instr = UniformInstrInfo::of(OpCode::MODLRFREG).with_common_comment("lrf add");
            instr.set_common_field(OperandField::Src0, index_field(add.tgt));
            let held = mod_reg_src1(
                add.tgt,
                *src0,
                *src1,
                &mut copy,
                (OpCode::LRFREGCOPY, "LRF <- LRF"),
            );
            instr.set_common_field(OperandField::Src1, index_field(held));
            instr
        }
    };
    ScalarUpdate::of(copy, vec![instr], refused)
}

/// WHAT A LAR OR EAR ADD ADDS — the two arms of `:826-871`, its `else` an `emitError`.
#[derive(Debug, Clone, PartialEq)]
pub enum AddrAddOperands {
    /// A register and an immediate, in either order — ⭐ ONE ARM: the reference takes whichever
    /// operand is not the immediate (`:827-830`).
    RegImm {
        /// `getValueRegIndex(reg_operand)`.
        reg: Option<RegIndex>,
        /// `imm_operand`'s defining op.
        imm: ImmSource,
    },
    /// Both operands in the file.
    RegReg {
        /// `getValueRegIndex(inp1)`.
        src0: Option<RegIndex>,
        /// `getValueRegIndex(inp2)`.
        src1: Option<RegIndex>,
    },
}

/// A LAR OR EAR ADD.
#[derive(Debug, Clone, PartialEq)]
pub struct AddrAdd {
    /// `getRegLocale()` — ⛔ THE OPERANDS' OWN LOCALE IS THIS ONE (`DT_CHECK` `:819-821`), so it is
    /// not stated a second time and cannot disagree.
    pub file: AddrFile,
    /// `getValueRegIndex(getResult())`.
    pub tgt: Option<RegIndex>,
    /// The two operands.
    pub operands: AddrAddOperands,
}

/// Replaces: e082_ConstructLARorEARADDInstr
///
/// The add that steps a local or external address register, and the copy it may need in front.
///
/// ⛔ `ADD*IMM` AND `MOD*REG` BOTH WRITE THE REGISTER THEY READ (`:838-841`, `:865-868`), so a third
/// target costs a `LARREGCOPY`/`EARREGCOPY`.
/// ⭐ THE IMMEDIATE ARM'S SCALE IS THE NON-IMMEDIATE OPERAND'S (`:834-836`), which is the component's
/// either way — the locale argument of `GetAddressScale` is dead.
#[must_use]
pub fn construct_lar_or_ear_add_instr<A: Arch>(
    add: &AddrAdd,
    comp: Component,
    element_size: Bits,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> ScalarUpdate {
    let held = |index| Reg {
        locale: add.file.locale(),
        index,
    };
    if full_reg_init {
        let mut regs = match &add.operands {
            AddrAddOperands::RegImm { reg, .. } => vec![held(*reg)],
            AddrAddOperands::RegReg { src0, src1 } => vec![held(*src0), held(*src1)],
        };
        regs.push(held(add.tgt));
        add_to_regs_to_init(units, &regs, regs_to_init);
    }
    let comment = format!("{} add", add.file.locale().spelling());
    let mut refused = Vec::new();
    let mut copy = None;
    let instr = match &add.operands {
        AddrAddOperands::RegImm { reg, imm } => {
            let mut instr = UniformInstrInfo::of(match add.file {
                AddrFile::Lar => OpCode::ADDLARIMM,
                AddrFile::Ear => OpCode::ADDEARIMM,
            })
            .with_common_comment(&comment);
            instr.set_common_field(OperandField::Src0, index_field(add.tgt));
            refused = fill_imm_field::<A>(
                &mut instr,
                OperandField::Imm,
                imm,
                comp,
                element_size,
                address_scale::<A>(comp),
            );
            if *reg != add.tgt {
                let (opcode, text) = add.file.copy();
                copy = Some(reg_copy(opcode, text, add.tgt, *reg));
            }
            instr
        }
        AddrAddOperands::RegReg { src0, src1 } => {
            let mut instr = UniformInstrInfo::of(match add.file {
                AddrFile::Lar => OpCode::MODLARREG,
                AddrFile::Ear => OpCode::MODEARREG,
            })
            .with_common_comment(&comment);
            instr.set_common_field(OperandField::Src0, index_field(add.tgt));
            let src1 = mod_reg_src1(add.tgt, *src0, *src1, &mut copy, add.file.copy());
            instr.set_common_field(OperandField::Src1, index_field(src1));
            instr
        }
    };
    ScalarUpdate::of(copy, vec![instr], refused)
}

/// ⛔ 2 MB — THE WIDEST `lrfimm` THERE IS, and live-range reduction or const propagation can hand
/// the sub up to 4 MB, which is why one op becomes two (`:1046-1049`).
const MAX_LRF_IMM: i64 = 0x1F_FFFF;

/// WHAT `imm - lrf` SUBTRACTS FROM — ⛔ A THIRD DEFINING OP EMITS NO SUB AT ALL (`:1039-1073` has no
/// `else`), which this closed pair makes unrepresentable.
#[derive(Debug, Clone, PartialEq)]
pub enum SubImm {
    /// `sentient.constant`.
    Constant(i64),
    /// `uniform.query_map`'s constant target values, one per unit and fold.
    Mapped(Vec<FoldConstant>),
}

/// WHAT AN LRF SUB SUBTRACTS — the two locale pairs of `:1029-1140`, its `else` an `emitError`.
#[derive(Debug, Clone, PartialEq)]
pub enum LrfSubOperands {
    /// `lrf - imm`.
    RegImm {
        /// `getValueRegIndex(inp1)`.
        src0: Option<RegIndex>,
        /// `inp2`'s defining op.
        imm: ImmSource,
    },
    /// `imm - lrf` — ⛔ LX-ONLY (`DT_CHECK_MSG` `:1042`), so its immediate field is always `lrfimm`.
    ImmReg {
        /// `inp1`'s defining op.
        imm: SubImm,
        /// `getValueRegIndex(inp2)`.
        src1: Option<RegIndex>,
    },
}

/// AN LRF SUB.
#[derive(Debug, Clone, PartialEq)]
pub struct LrfSub {
    /// `getValueRegIndex(getResult())`.
    pub tgt: Option<RegIndex>,
    /// The two operands.
    pub operands: LrfSubOperands,
}

/// Replaces: e083_ConstructLRFSUBInstr
///
/// The sub that steps an LRF pointer, splitting an immediate too wide for one field into a sub and a
/// following modify.
///
/// ⛔ THE SPLIT IS PER UNIT (`:1091-1101`): a map whose largest value is over range clamps only the
/// units that are over and gives every other unit a remainder of ZERO, so they all run both ops.
/// ⚠️ AND `imm - lrf`'s SCALING IS FLOATING POINT AND UNWRAPPED where the `lrf - imm` arm goes through
/// [`fill_imm_field`] (`:1045` against `:4136-4141`), so one immediate can round two ways.
#[must_use]
pub fn construct_lrf_sub_instr<A: Arch>(
    sub: &LrfSub,
    comp: Component,
    element_size: Bits,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> ScalarUpdate {
    let lrf = |index| Reg {
        locale: SenRegType::Lrf,
        index,
    };
    if full_reg_init {
        let mut regs = match &sub.operands {
            LrfSubOperands::RegImm { src0, .. } => vec![lrf(*src0)],
            LrfSubOperands::ImmReg { src1, .. } => vec![lrf(*src1)],
        };
        regs.push(lrf(sub.tgt));
        add_to_regs_to_init(units, &regs, regs_to_init);
    }
    let scale = address_scale::<A>(comp);
    let imm_field = lrf_imm_field(comp);
    let mut refused = Vec::new();
    let mut copy = None;
    let instrs = match &sub.operands {
        LrfSubOperands::RegImm { src0, imm } => {
            let mut instr = UniformInstrInfo::of(OpCode::MODLRFIMM).with_common_comment("lrf sub");
            refused = fill_imm_field::<A>(&mut instr, imm_field, imm, comp, element_size, scale);
            instr.set_common_field(OperandField::Src0, index_field(sub.tgt));
            if *src0 != sub.tgt {
                copy = Some(lrf_copy(sub.tgt, *src0));
            }
            vec![instr]
        }
        LrfSubOperands::ImmReg { imm, src1 } => {
            let mut instr = UniformInstrInfo::of(OpCode::SUBLRFIMM).with_common_comment("lrf sub");
            instr.set_common_field(OperandField::Src0, index_field(sub.tgt));
            // ⭐ THE MODIFY THAT CARRIES THE REMAINDER HAS NO COMMENT OF ITS OWN (`:1055-1058`).
            let mut rest = UniformInstrInfo::of(OpCode::MODLRFIMM);
            rest.set_common_field(OperandField::Src0, index_field(sub.tgt));
            let widen = f64::from(element_size.0) / 8.0 / f64::from(scale.get());
            let instrs = match imm {
                SubImm::Constant(value) => {
                    let imm = (*value as f64 * widen) as i64;
                    if imm > MAX_LRF_IMM {
                        instr.set_common_field(imm_field, int(MAX_LRF_IMM));
                        rest.set_common_field(imm_field, int(imm - MAX_LRF_IMM));
                        vec![instr, rest]
                    } else {
                        instr.set_common_field(imm_field, int(imm));
                        vec![instr]
                    }
                }
                SubImm::Mapped(entries) => {
                    let scaled: Vec<FoldConstant> = entries
                        .iter()
                        .map(|entry| FoldConstant {
                            value: (entry.value as f64 * widen) as i64,
                            ..*entry
                        })
                        .collect();
                    if scaled.iter().any(|entry| entry.value > MAX_LRF_IMM) {
                        let remain: Vec<FoldConstant> = scaled
                            .iter()
                            .map(|entry| FoldConstant {
                                value: (entry.value - MAX_LRF_IMM).max(0),
                                ..*entry
                            })
                            .collect();
                        let clamped: Vec<FoldConstant> = scaled
                            .iter()
                            .map(|entry| FoldConstant {
                                value: entry.value.min(MAX_LRF_IMM),
                                ..*entry
                            })
                            .collect();
                        refused = folding_needed(
                            instr.add_const_entries_to_operand_map(imm_field, &clamped, 1.0, None),
                        );
                        refused.extend(folding_needed(
                            rest.add_const_entries_to_operand_map(imm_field, &remain, 1.0, None),
                        ));
                        vec![instr, rest]
                    } else {
                        refused = folding_needed(
                            instr.add_const_entries_to_operand_map(imm_field, &scaled, 1.0, None),
                        );
                        vec![instr]
                    }
                }
            };
            if *src1 != sub.tgt {
                copy = Some(lrf_copy(sub.tgt, *src1));
            }
            instrs
        }
    };
    ScalarUpdate::of(copy, instrs, refused)
}

/// AN `imm - lar` — ⛔ THE ONLY SUB THIS FUNCTION EMITS.
#[derive(Debug, Clone, PartialEq)]
pub struct AddrSub {
    /// `getValueRegIndex(getResult())`.
    pub tgt: Option<RegIndex>,
    /// `inp1`'s defining op — the immediate is the LEFT operand.
    pub imm: ImmSource,
    /// `getValueRegIndex(inp2)`.
    pub src1: Option<RegIndex>,
}

/// Replaces: e084_ConstructLARorEARSUBInstr
///
/// `imm - lar`, with the copy that puts the source in the target register first.
///
/// ⛔ DESPITE ITS NAME THERE IS NO EAR PATH (`:1153-1178`): `ear - imm`, `imm - ear` and every LAR
/// order but this one reach the `emitError`, so `SUBLARIMM` and `LARREGCOPY` are the whole emission.
/// ⭐ THE OPERAND'S LOCALE IS `getRegLocale()` BY ITS `DT_CHECK_MSG` (`:1147`), so `lar` is also the
/// comment.
#[must_use]
pub fn construct_lar_or_ear_sub_instr<A: Arch>(
    sub: &AddrSub,
    comp: Component,
    element_size: Bits,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> ScalarUpdate {
    let lar = |index| Reg {
        locale: SenRegType::Lar,
        index,
    };
    if full_reg_init {
        add_to_regs_to_init(units, &[lar(sub.src1), lar(sub.tgt)], regs_to_init);
    }
    let mut instr = UniformInstrInfo::of(OpCode::SUBLARIMM).with_common_comment("lar sub");
    instr.set_common_field(OperandField::Src0, index_field(sub.tgt));
    let refused = fill_imm_field::<A>(
        &mut instr,
        OperandField::Imm,
        &sub.imm,
        comp,
        element_size,
        address_scale::<A>(comp),
    );
    let copy = (sub.src1 != sub.tgt)
        .then(|| reg_copy(OpCode::LARREGCOPY, "LAR <- LAR", sub.tgt, sub.src1));
    ScalarUpdate::of(copy, vec![instr], refused)
}

/// `getValueRegIndex`'s answer as a field value — ⛔ THE REFERENCE WRITES ITS `-1` STRAIGHT INTO THE
/// FIELD (`SentientOps.cpp:1886`), so an unassigned register is a negative index in the program.
fn index_field(index: Option<RegIndex>) -> Operand {
    int(index.map_or(-1, |at| i64::from(at.get())))
}

/// `setOperandMap(OperandMap(…))` — ⛔ IT REPLACES THE MAP (`UniformInstrAndBlock.hpp:133`): every
/// per-unit field the instruction already carried is dropped rather than merged.
fn map_into(
    instr: &mut UniformInstrInfo,
    field: OperandField,
    entries: &[MappedEntry],
    mode: MapMode,
    scale: f64,
) -> Vec<OperandMapRefusal> {
    let mapped = add_entry_to_operand_map(entries, mode, scale);
    instr.operand_map = mapped
        .value
        .map_or_else(Vec::new, |value| vec![(field, value)]);
    mapped.refused
}

/// WHAT A `sentient.for`'s TRIP COUNT IS — the four defining ops of `:58-107`.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopBound {
    /// `sentient.constant` — one trip count for every unit.
    Constant(i64),
    /// `uniform.query_map` — one per unit; ⛔ MODE `none` AND SCALE 1, `OperandMap`'s own defaults
    /// (`UniformInstrAndBlock.hpp:38-39`), which this call site leaves unstated.
    Mapped(Vec<MappedEntry>),
    /// `symbol.create_symbol` — ⛔ *"No symbol id present in the loop bound"* IS UNREPRESENTABLE: the
    /// id is this variant's payload.
    Symbol(i64),
    /// Anything else — a count already in a `jcr`, which makes this a DYNAMIC loop that tests its
    /// counter and jumps past its own body instead of counting down.
    Dynamic {
        /// The `jcr` the count is in.
        src0: RegIndex,
        /// The op after the loop, whose label this instruction jumps to.
        next: OpSite,
    },
}

/// AN `MVLOOPCNT` AND WHAT ITS PER-UNIT TRIP COUNT COULD NOT HOLD.
#[derive(Debug, Clone, PartialEq)]
pub struct MvLoop {
    /// The instruction.
    pub instr: UniformInstrInfo,
    /// The offenders, in entry order.
    pub refused: Vec<OperandMapRefusal>,
}

/// Replaces: e042_ConstructMVLoopInstr
///
/// The loop-count move that opens a `sentient.for`: a trip count in `imm`, or a dynamic bound in a
/// `jcr` plus the exit label to jump to.
///
/// ⛔ THE DYNAMIC ARM READS THE LABEL MAP BACK (`:101-104`) — a loop whose successor is already
/// labelled jumps to THAT label, and mints its own only as a fallback.
/// ⭐ AND ITS `dbgName` GOES INSIDE THE LABEL, where the other three arms put it in the comment.
#[must_use]
pub fn construct_mv_loop_instr(
    bound: &LoopBound,
    lccr: RegIndex,
    dbg_name: Option<&str>,
    labels: &mut Labels,
    counter: &mut LabelCounter,
) -> MvLoop {
    let mut instr = UniformInstrInfo::of(OpCode::MVLOOPCNT);
    let counted = format!("for-loop-imm-lccr-{}", lccr.get());
    let mut refused = Vec::new();
    match bound {
        LoopBound::Constant(value) => {
            instr = instr.with_common_comment(dbg_name.unwrap_or(&counted));
            instr.set_common_field(OperandField::Imm, int(*value));
        }
        LoopBound::Mapped(entries) => {
            instr = instr.with_common_comment(dbg_name.unwrap_or(&counted));
            refused = map_into(&mut instr, OperandField::Imm, entries, MapMode::None, 1.0);
        }
        LoopBound::Symbol(id) => {
            instr = instr.with_common_comment(dbg_name.unwrap_or(&counted));
            instr.set_common_field(OperandField::Imm, variable_symbol(*id));
        }
        LoopBound::Dynamic { src0, next } => {
            instr.set_common_field(OperandField::DynLoop, int(1));
            // ⭐ LOWERCASE HERE, where the `if` lowering spells the same register `JCR3`.
            instr.set_common_field(
                OperandField::Src0,
                descriptive(&format!("jcr{}", src0.get())),
            );
            let label = match dbg_name {
                Some(name) => format!("for-loop-jcr-{}({name})", counter.bump()),
                None => format!("for-loop-jcr-{}", counter.bump()),
            };
            instr = instr.with_common_comment(&format!("{label}-begin"));
            let end = labels.claim(*next, format!("{label}-end")).to_owned();
            instr.set_common_field(OperandField::PcTarget, instr_tag(&end));
        }
    }
    MvLoop { instr, refused }
}

/// WHICH ADDRESS FILE AN ASSIGNMENT TOUCHES — `lar` or `ear`, which picks the opcode and the comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrFile {
    /// The local address file.
    Lar,
    /// The external address file.
    Ear,
}

/// WHAT A `JIMMCOPY`'s IMMEDIATE IS (`:134-151`).
#[derive(Debug, Clone, PartialEq)]
pub enum JcrImm {
    /// `sentient.constant`.
    Constant(i64),
    /// `uniform.query_map` — mode `none`, scale 1.
    Mapped(Vec<MappedEntry>),
    /// `symbol.create_symbol` — the id, as a `VARIABLE_SYMBOL`.
    Symbol(i64),
}

/// WHAT AN `IMMCOPY`'s IMMEDIATE IS (`:167-196`).
///
/// ⭐ A `uniform.uniformize_regions` SOURCE IS REACHED BY RE-ENTERING THE WHOLE FUNCTION ON THE
/// REGION'S YIELDED VALUE (`:176-180`): that walk is operand-reaching and lands back in this same
/// arm, so what arrives here is the immediate at the end of the chain.
#[derive(Debug, Clone, PartialEq)]
pub enum LrfImm {
    /// `sentient.constant`, scaled into the target's address granularity.
    Constant(i64),
    /// `uniform.query_map`, scaled the same way.
    Mapped(Vec<MappedEntry>),
    /// `symbol.create_symbol` — ⛔ UNSCALED, the id going in whole.
    Symbol(i64),
}

/// WHAT A `LARIMM`/`EARIMM`'s IMMEDIATE IS (`:239-275`).
#[derive(Debug, Clone, PartialEq)]
pub enum AddrImm {
    /// `sentient.constant`, scaled.
    Constant(i64),
    /// `dataflow.get_unit` — its core id, unscaled.
    Core(Core),
    /// `uniform.query_map` over constants — mode `none`, scaled.
    MappedConstants(Vec<MappedEntry>),
    /// `uniform.query_map` over `dataflow.get_unit`s — mode `unit_id`, and ⛔ UNSCALED.
    MappedUnits(Vec<MappedEntry>),
    /// `symbol.create_symbol`.
    Symbol(i64),
}

/// WHAT A `GTRIMM`'s IMMEDIATE IS (`:288-299`).
///
/// ⭐ THE REFERENCE HAS NO `else` HERE: a third kind of source leaves the `GTRIMM` carrying no
/// immediate at all, which this enum makes unwritable.
#[derive(Debug, Clone, PartialEq)]
pub enum GtrImm {
    /// `uniform.query_map` — mode `none`, scale 1.
    Mapped(Vec<MappedEntry>),
    /// `dataflow.create_multicast_group`, ALREADY ENCODED — `encodeMulticastGroupInfo` is outside this
    /// campaign's 130 — whose decode goes into the comment.
    Multicast {
        /// `encodeMulticastGroupInfo(op)`.
        encoded: i64,
        /// `decodeMulticastGroupInfo(encoded)`.
        decoded: String,
    },
}

/// WHICH TWO FILES AN ASSIGNMENT COPIES BETWEEN — the arms of `:130-308`.
///
/// ⛔ `LBR`/`EBR <- IMM` IS UNREPRESENTABLE, which is the reference's own `DT_CHECK` (`:127-129`).
#[derive(Debug, Clone, PartialEq)]
pub enum AssignKind {
    /// `imm -> jcr`: `JIMMCOPY`.
    JcrFromImm(JcrImm),
    /// `jcr -> jcr`: a `JADD` of zero.
    JcrFromJcr,
    /// `imm -> lrf`: `IMMCOPY`.
    LrfFromImm(LrfImm),
    /// `lrf -> lrf`: `LRFREGCOPY`.
    LrfFromLrf,
    /// `imm -> xrfrdptr`/`xrfwrptr`: `XRFACCESS` with the pointer `set` rather than stepped.
    XrfPtrFromImm {
        /// Which pointer is set.
        ptr: XrfPtr,
        /// The index it is set to, before the row wraparound.
        val: i64,
    },
    /// `imm -> lar`/`ear`: `LARIMM`/`EARIMM`.
    AddrFromImm {
        /// Which file.
        file: AddrFile,
        /// The immediate.
        imm: AddrImm,
    },
    /// `lar -> lar` or `ear -> ear`: `LARREGCOPY`/`EARREGCOPY` — ⛔ NEVER BETWEEN THE TWO FILES.
    AddrFromAddr(AddrFile),
    /// `imm -> gtr`: `GTRIMM`.
    GtrFromImm(GtrImm),
}

/// ONE IMPLICIT ASSIGNMENT — a `sentient.copy`, or any lowering that moves a value into a register.
#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    /// `getValueRegIndex(src)` — ⛔ `None` IS THE REFERENCE'S `-1`: an immediate, or a register the
    /// allocator never assigned (`SentientOps.cpp:1886`).
    pub src_index: Option<RegIndex>,
    /// `getValueRegIndex(tgt)`.
    pub tgt_index: Option<RegIndex>,
    /// Which files, and what the source holds.
    pub kind: AssignKind,
}

/// AN ASSIGNMENT'S INSTRUCTION, ITS COPY-OP COUNT AND WHAT ITS PER-UNIT MAP COULD NOT HOLD.
#[derive(Debug, Clone, PartialEq)]
pub struct Assigned {
    /// ⛔ `None` WHERE THE REFERENCE ANSWERS `std::nullopt`: a copy into the register the value is
    /// already in, or an immediate the program header will carry instead.
    pub instr: Option<UniformInstrInfo>,
    /// `++num_copy_ops_` — 1 for each of the six copy opcodes (`:309-313`).
    pub copy_ops: CopyOps,
    /// The offenders, in entry order.
    pub refused: Vec<OperandMapRefusal>,
}

impl Assigned {
    /// `std::nullopt` — no instruction, nothing counted, nothing refused.
    fn none() -> Assigned {
        Assigned {
            instr: None,
            copy_ops: CopyOps(0),
            refused: Vec::new(),
        }
    }
}

/// Replaces: e043_ConstructAssignInstr
///
/// The copy between two register files, with the immediate scaled into the target's granularity.
///
/// ⛔ A COPY INTO THE REGISTER THE VALUE IS ALREADY IN IS NO INSTRUCTION (`:125`), which also makes
/// the reference's same-locale arm (`:226-227`) dead code.
/// ⛔ `program_header` DIVERTS ONLY THE TWO IMMEDIATE ARMS (`:160,230`).
/// ⛔ AND A DIRECT LAR/EAR IMMEDIATE WRITES `imm` LITERALLY WHERE ITS MAPPED TWIN WRITES
/// `imm_field_name` — `lrfimm` on an LX component (`:242,252`).
#[must_use]
pub fn construct_assign_instr<A: Arch>(
    assign: &Assign,
    comp: Component,
    element_size: Bits,
    program_header: bool,
) -> Assigned {
    if assign.src_index == assign.tgt_index {
        return Assigned::none();
    }
    let imm_field = match comp {
        Component::Lxlu | Component::Lxsu => OperandField::Lrfimm,
        _ => OperandField::Imm,
    };
    let scale = f64::from(element_size.0) / 8.0 / f64::from(address_scale::<A>(comp).get());
    let mut refused = Vec::new();
    let instr = match &assign.kind {
        AssignKind::JcrFromImm(imm) => {
            let mut instr =
                UniformInstrInfo::of(OpCode::JIMMCOPY).with_common_comment("JCR <- IMM");
            instr.set_common_field(OperandField::JcrTarget, index_field(assign.tgt_index));
            match imm {
                JcrImm::Constant(value) => {
                    instr.set_common_field(OperandField::Imm, int(*value));
                }
                JcrImm::Mapped(entries) => {
                    refused = map_into(&mut instr, OperandField::Imm, entries, MapMode::None, 1.0);
                }
                JcrImm::Symbol(id) => {
                    instr.set_common_field(OperandField::Imm, variable_symbol(*id));
                }
            }
            instr
        }
        AssignKind::JcrFromJcr => {
            let mut instr = UniformInstrInfo::of(OpCode::JADD).with_common_comment("JCR <- JCR");
            instr.set_common_field(OperandField::JcrSelect, int(1));
            instr.set_common_field(OperandField::JcrTarget, index_field(assign.tgt_index));
            instr.set_common_field(OperandField::Src0, index_field(assign.src_index));
            instr.set_common_field(OperandField::Imm, int(0));
            instr
        }
        AssignKind::LrfFromImm(imm) => {
            if program_header {
                return Assigned::none();
            }
            let mut instr = UniformInstrInfo::of(OpCode::IMMCOPY).with_common_comment("LRF <- IMM");
            instr.set_common_field(OperandField::Src0, index_field(assign.tgt_index));
            match imm {
                LrfImm::Constant(value) => {
                    let imm = (*value as f64 * scale) as i64;
                    instr.set_common_field(imm_field, int(imm));
                }
                LrfImm::Mapped(entries) => {
                    refused = map_into(&mut instr, imm_field, entries, MapMode::None, scale);
                }
                LrfImm::Symbol(id) => {
                    instr.set_common_field(imm_field, variable_symbol(*id));
                }
            }
            instr
        }
        AssignKind::LrfFromLrf => {
            let mut instr =
                UniformInstrInfo::of(OpCode::LRFREGCOPY).with_common_comment("LRF <- LRF");
            instr.set_common_field(OperandField::Src0, index_field(assign.tgt_index));
            instr.set_common_field(OperandField::Src1, index_field(assign.src_index));
            instr
        }
        AssignKind::XrfPtrFromImm { ptr, val } => {
            let (comment, moved, imm, held) = match ptr {
                XrfPtr::Read => (
                    "set xrf rd index",
                    OperandField::RdptrUpd,
                    OperandField::RdptrImm,
                    OperandField::WrptrUpd,
                ),
                XrfPtr::Write => (
                    "set xrf wr index",
                    OperandField::WrptrUpd,
                    OperandField::WrptrImm,
                    OperandField::RdptrUpd,
                ),
            };
            let mut instr = UniformInstrInfo::of(OpCode::XRFACCESS).with_common_comment(comment);
            instr.set_common_field(moved, descriptive("set"));
            instr.set_common_field(held, descriptive("no"));
            instr.set_common_field(imm, int(addr_wraparounded::<A>(*val, AddrSpace::PtXrf)));
            instr
        }
        AssignKind::AddrFromImm { file, imm } => {
            if program_header {
                return Assigned::none();
            }
            let (opcode, comment) = match file {
                AddrFile::Lar => (OpCode::LARIMM, "LAR <- IMM"),
                AddrFile::Ear => (OpCode::EARIMM, "EAR <- IMM"),
            };
            let mut instr = UniformInstrInfo::of(opcode).with_common_comment(comment);
            instr.set_common_field(OperandField::Src0, index_field(assign.tgt_index));
            match imm {
                AddrImm::Constant(value) => {
                    let imm = (*value as f64 * scale) as i64;
                    instr.set_common_field(OperandField::Imm, int(imm));
                }
                AddrImm::Core(core) => {
                    instr.set_common_field(OperandField::Imm, int(i64::from(core.get())));
                }
                AddrImm::MappedConstants(entries) => {
                    refused = map_into(&mut instr, imm_field, entries, MapMode::None, scale);
                }
                AddrImm::MappedUnits(entries) => {
                    refused = map_into(&mut instr, imm_field, entries, MapMode::UnitId, 1.0);
                }
                AddrImm::Symbol(id) => {
                    instr.set_common_field(OperandField::Imm, variable_symbol(*id));
                }
            }
            instr
        }
        AssignKind::AddrFromAddr(file) => {
            let (opcode, comment) = match file {
                AddrFile::Lar => (OpCode::LARREGCOPY, "LAR <- LAR"),
                AddrFile::Ear => (OpCode::EARREGCOPY, "EAR <- EAR"),
            };
            let mut instr = UniformInstrInfo::of(opcode).with_common_comment(comment);
            instr.set_common_field(OperandField::Src0, index_field(assign.tgt_index));
            instr.set_common_field(OperandField::Src1, index_field(assign.src_index));
            instr
        }
        AssignKind::GtrFromImm(imm) => {
            // ⛔ THE TRAILING SPACE IS THE REFERENCE'S (`:287`), and only the mapped arm keeps it.
            let mut instr = UniformInstrInfo::of(OpCode::GTRIMM).with_common_comment("GTR <- IMM ");
            instr.set_common_field(OperandField::Src0, index_field(assign.tgt_index));
            match imm {
                GtrImm::Mapped(entries) => {
                    refused = map_into(&mut instr, imm_field, entries, MapMode::None, 1.0);
                }
                GtrImm::Multicast { encoded, decoded } => {
                    instr.set_common_field(OperandField::Imm, int(*encoded));
                    instr = instr.with_common_comment(&format!("GTR <- IMM: {decoded}"));
                }
            }
            instr
        }
    };
    // `:309-313` — ⭐ `LRFCOPY` IS IN THE SET AND NO ARM HERE EMITS IT.
    let counted = matches!(
        instr.opcode,
        OpCode::EARREGCOPY
            | OpCode::LARREGCOPY
            | OpCode::IMMCOPY
            | OpCode::JIMMCOPY
            | OpCode::LRFCOPY
            | OpCode::LRFREGCOPY
    );
    Assigned {
        instr: Some(instr),
        copy_ops: CopyOps(u32::from(counted)),
        refused,
    }
}

/// Replaces: e044_ConstructBranchExitInstr
///
/// The NOP that closes a branch — ⭐ `be:be`, a field whose value is its own name.
#[must_use]
pub fn construct_branch_exit_instr() -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::NOP).with_common_comment("Branch End");
    instr.set_common_field(OperandField::Be, descriptive("be"));
    instr
}

/// THE TILE AN IMPLICIT SYNC BOUNDS — `implicit_sync_memory_boundary`, written straight into
/// `tilesize` with no conversion (`:409`), so its unit is the ISA field's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryTile(pub i32);

/// `lx_consumer_map` (`:356-358`) — ⛔ A CONSUMER THE MAP DOES NOT NAME CONTRIBUTES 0, which is what
/// `std::map::operator[]` default-inserts for it.
const fn lx_bit(peer: Consumer) -> i64 {
    match peer {
        Consumer::L3lu => 4,
        Consumer::L3su => 8,
        Consumer::Lxlu => 16,
        Consumer::Lxsu => 32,
        Consumer::LxluN => 64,
        Consumer::LxsuN => 128,
        _ => 0,
    }
}

/// `l3_consumer_map` (`:359-361`) — ⭐ THE SAME FOUR BITS NAME THE PER-CORELET LX UNITS HERE.
const fn l3_bit(peer: Consumer) -> i64 {
    match peer {
        Consumer::L3lu => 4,
        Consumer::L3su => 8,
        Consumer::Lxlu0 => 16,
        Consumer::Lxsu0 => 32,
        Consumer::Lxlu1 => 64,
        Consumer::Lxsu1 => 128,
        _ => 0,
    }
}

/// WHICH UNITS A `sentient.sync` WAITS ON, AND FROM WHERE — the three component groups of `:373-397`
/// plus the L0's implicit-boundary case.
///
/// ⭐ THE REFERENCE'S FALL-THROUGH (a component in none of the three groups) EMITS WHAT [`Sync::Lx`]
/// WITH NO PEERS DOES, and its only caller admits just these six (`LowerSentientHelper.cpp:358`).
/// ⛔ THE PEERS ARE [`Consumer`]s AND NOT THE ISLAND'S `SyncHalf`: the vendor's own programs name the
/// wildcard and per-corelet consumers (`lxsuN`, `lxlu0`) that a `rendezvous` cannot mint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sync {
    /// An `LXLU`/`LXSU` sync — [`lx_bit`].
    Lx {
        /// `$mode`.
        mode: SyncMode,
        /// `$units`.
        peers: Vec<Consumer>,
    },
    /// An `L3LU` sync — [`l3_bit`], and ⛔ THE ONLY COMPONENT A SOFT SYNC IS ALLOWED ON (`:392`).
    L3lu {
        /// `$mode`.
        mode: SyncMode,
        /// `$units`.
        peers: Vec<Consumer>,
        /// `$soft`.
        soft: bool,
    },
    /// An `L3SU` sync — the same bits, never soft.
    L3su {
        /// `$mode`.
        mode: SyncMode,
        /// `$units`.
        peers: Vec<Consumer>,
    },
    /// An `L0LU`/`L0SU` sync — ⭐ NO CONSUMER BITS AT ALL: the `units` attribute is not read.
    L0 {
        /// `$mode`.
        mode: SyncMode,
    },
    /// An `L0LU`/`L0SU` sync that sets the implicit memory boundary — ⛔ ITS MODE IS `sendrecv` BY
    /// CONSTRUCTION (`:404`) and ⛔ IT WRITES NO `synctag` AT ALL (`:423-424`).
    L0Implicit {
        /// `implicit_sync_memory_boundary`.
        tile: BoundaryTile,
    },
}

/// Replaces: e045_ConstructSyncInstr
///
/// The rendezvous: a mode and one bit per peer in `synctag`, or the L0's implicit tile boundary.
///
/// ⛔ AN IMPLICIT SYNC RETURNS BEFORE `synctag` IS SET (`:424`) — its tag is the tile, not the peers.
/// ⭐ THE `soft` ARCH GATE IS VACUOUS HERE: `RCUDD1A` is the oldest generation this crate models.
#[must_use]
pub fn construct_sync_instr<A: Arch>(
    sync: &Sync,
    l0_tethered: bool,
    dbg_name: Option<&str>,
) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::SYNC);
    let mode = match sync {
        Sync::Lx { mode, .. }
        | Sync::L3lu { mode, .. }
        | Sync::L3su { mode, .. }
        | Sync::L0 { mode } => *mode,
        Sync::L0Implicit { .. } => SyncMode::SendRecv,
    };
    let mut synctag = match mode {
        SyncMode::Send => 1,
        SyncMode::Recv => 2,
        SyncMode::SendRecv => 3,
    };
    let mut comment = mode.spelling().to_owned();
    // `:410-417,426-433` — ⭐ SEN1P5 NEEDS A DESTINATION ON EVERY L0 SYNC, implicit or not.
    let syncdest = |instr: &mut UniformInstrInfo| {
        if A::GEN >= IsaGen::Sen1p5 {
            let dest = if l0_tethered { "both" } else { "self" };
            instr.set_common_field(OperandField::Syncdest, descriptive(dest));
        }
    };
    match sync {
        Sync::Lx { peers, .. } => {
            for peer in peers {
                comment.push('-');
                comment.push_str(peer.spelling());
                synctag += lx_bit(*peer);
            }
        }
        Sync::L3lu { peers, soft, .. } => {
            for peer in peers {
                comment.push('-');
                comment.push_str(peer.spelling());
                synctag += l3_bit(*peer);
            }
            if *soft {
                comment.push_str("-soft");
                instr.set_common_field(OperandField::Soft, descriptive("yes"));
            }
        }
        Sync::L3su { peers, .. } => {
            for peer in peers {
                comment.push('-');
                comment.push_str(peer.spelling());
                synctag += l3_bit(*peer);
            }
        }
        Sync::L0 { .. } => syncdest(&mut instr),
        Sync::L0Implicit { tile } => {
            comment.push_str("-implicit");
            instr.set_common_field(OperandField::Implicit, descriptive("yes"));
            instr.set_common_field(OperandField::Tilesize, int(i64::from(tile.0)));
            syncdest(&mut instr);
            let comment = dbg_name.map_or_else(|| format!("sync {comment}"), str::to_owned);
            return instr.with_common_comment(&comment);
        }
    }
    instr.set_common_field(OperandField::Synctag, int(synctag));
    let comment = dbg_name.map_or_else(|| format!("sync {comment}"), str::to_owned);
    instr.with_common_comment(&comment)
}

/// Replaces: e046_ConstructJMPInstr
///
/// The unconditional jump: `mode:always` and the caller's label in `pc_target`.
/// ⭐ THE INSTRUCTION `createJmpInstr` BUILDS (`UniformInstrAndBlock.cpp:30-41`), except that the
/// target here is a label something else defined rather than a number drawn for it.
#[must_use]
pub fn construct_jmp_instr(target: &str) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::JCMP).with_common_comment("jump");
    instr.set_common_field(OperandField::Mode, descriptive("always"));
    instr.set_common_field(OperandField::PcTarget, instr_tag(target));
    instr
}

/// WHAT A `sentient.if` COMPARES ITS REGISTER AGAINST (`:608-624,667-675`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpImm {
    /// `sentient.constant` — the value, into `src1`.
    Constant(i64),
    /// `symbol.create_symbol` — ⛔ ITS ID INTO `src1` AS A `VARIABLE_SYMBOL`, and the id's absence is
    /// unrepresentable where the reference reports it (`:616-619`).
    Symbol(i64),
}

/// WHICH REGISTERS A `sentient.if` COMPARES — the seven operand arms of `:627-662`.
///
/// ⛔ WHICH SIDE THE IMMEDIATE IS ON IS LOAD-BEARING: it decides the inverted mode, and only an
/// immediate on the RIGHT changes it — ⭐ the reference's `lhs == imm` and neither-is-imm branches
/// agree in all four signed predicates (`:493-545`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOperands {
    /// `imm CMP jcr`.
    ImmVsJcr {
        /// The immediate, on the left.
        imm: CmpImm,
        /// The `jcr`, which becomes `src0`.
        jcr: RegIndex,
    },
    /// `jcr CMP imm`.
    JcrVsImm {
        /// The `jcr`, which becomes `src0`.
        jcr: RegIndex,
        /// The immediate, on the right.
        imm: CmpImm,
    },
    /// `imm CMP lccr`.
    ImmVsLccr {
        /// The immediate, on the left.
        imm: CmpImm,
        /// The `lccr`, which becomes `src0`.
        lccr: RegIndex,
    },
    /// `lccr CMP imm`.
    LccrVsImm {
        /// The `lccr`, which becomes `src0`.
        lccr: RegIndex,
        /// The immediate, on the right.
        imm: CmpImm,
    },
    /// An `lccr` against a `jcr`, EITHER WAY ROUND — ⭐ THE REFERENCE'S TWO ARMS EMIT THE SAME
    /// INSTRUCTION (`:645-652`): `src0` is the LCCR and `src1` the JCR whichever side each was on,
    /// and with no immediate the mode does not move either.
    LccrAndJcr {
        /// `src0`.
        lccr: RegIndex,
        /// `src1`.
        jcr: RegIndex,
    },
    /// `jcr CMP jcr` — ⛔ `src0` IS THE RIGHT-HAND ONE (`:653-657`).
    JcrVsJcr {
        /// The left-hand `jcr`, which becomes `src1`.
        lhs: RegIndex,
        /// The right-hand `jcr`, which becomes `src0`.
        rhs: RegIndex,
    },
}

/// WHAT A `sentient.if` DOES WHEN THE CONDITION FAILS — its `else` region, as the jump sees it.
#[derive(Debug, Clone, PartialEq)]
pub enum ElseRegion {
    /// No `else` region at all (`:603-606`).
    None,
    /// An `else` region and its first op.
    Head {
        /// The op the `-else` label would be claimed for.
        at: OpSite,
        /// ⛔ `Some` ONLY WHERE THAT FIRST OP IS A `sentient.yield` (`:566`), carrying the registers it
        /// yields: an else that only yields what the `if` already holds needs no label at all.
        yields: Option<Vec<Reg>>,
    },
}

/// Replaces: e047_ConstructJCMPInstr
///
/// The conditional jump a `sentient.if` becomes: the INVERTED predicate, because the branch is taken
/// when the condition fails, and the label of the `else` body or of the op after the `if`.
///
/// ⛔ THE END LABEL IS NOT READ BACK (`:558-560`), unlike [`construct_mv_loop_instr`]'s: a successor
/// that already carries a label keeps it, and `pc_target` still names the one minted here.
/// ⛔ AN `else` THAT YIELDS EXACTLY THE `if`'s OWN RESULTS IS JUMPED PAST, not into (`:566-598`).
#[must_use]
pub fn construct_jcmp_instr(
    predicate: CmpPredicate,
    operands: CmpOperands,
    else_region: &ElseRegion,
    results: &[Reg],
    next: OpSite,
    labels: &mut Labels,
    counter: &mut LabelCounter,
) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::JCMP);
    let rhs_is_imm = matches!(
        operands,
        CmpOperands::JcrVsImm { .. } | CmpOperands::LccrVsImm { .. }
    );
    // `:487-549` — the inversion, which the immediate's side flips back for.
    let mode = match predicate {
        CmpPredicate::Eq => "ne",
        CmpPredicate::Ne => "eq",
        CmpPredicate::Slt if rhs_is_imm => "le",
        CmpPredicate::Slt => "ge",
        CmpPredicate::Sle if rhs_is_imm => "lt",
        CmpPredicate::Sle => "gt",
        CmpPredicate::Sgt if rhs_is_imm => "ge",
        CmpPredicate::Sgt => "le",
        CmpPredicate::Sge if rhs_is_imm => "gt",
        CmpPredicate::Sge => "lt",
    };
    instr.set_common_field(OperandField::Mode, descriptive(mode));

    let label = format!("if-label-{}", counter.bump());
    let end_label = format!("{label}-end");
    labels.claim(next, end_label.clone());
    let target = match else_region {
        ElseRegion::None => end_label,
        ElseRegion::Head { at, yields } => {
            // `:567-586` — a yield of the `if`'s own results generates no instructions to jump to.
            let set_else_tag = match yields {
                None => true,
                Some(yielded) if yielded.is_empty() => false,
                Some(yielded) => {
                    yielded.len() != results.len()
                        || yielded.iter().zip(results).any(|(from, to)| from != to)
                }
            };
            if set_else_tag {
                let else_label = format!("{label}-else");
                labels.claim(*at, else_label.clone());
                else_label
            } else {
                end_label
            }
        }
    };
    instr.set_common_field(OperandField::PcTarget, instr_tag(&target));

    let (use_jcr, src0, src1) = match operands {
        CmpOperands::ImmVsJcr { imm, jcr } | CmpOperands::JcrVsImm { jcr, imm } => {
            (true, format!("JCR{}", jcr.get()), Err(imm))
        }
        CmpOperands::ImmVsLccr { imm, lccr } | CmpOperands::LccrVsImm { lccr, imm } => {
            (false, format!("LCCR{}", lccr.get()), Err(imm))
        }
        CmpOperands::LccrAndJcr { lccr, jcr } => (
            false,
            format!("LCCR{}", lccr.get()),
            Ok(format!("JCR{}", jcr.get())),
        ),
        CmpOperands::JcrVsJcr { lhs, rhs } => (
            true,
            format!("JCR{}", rhs.get()),
            Ok(format!("JCR{}", lhs.get())),
        ),
    };
    instr.set_common_field(OperandField::Usejcr, boolean(use_jcr));
    instr.set_common_field(OperandField::Src0, descriptive(&src0));
    match src1 {
        Ok(register) => {
            instr.set_common_field(OperandField::Isimm, boolean(false));
            instr.set_common_field(OperandField::Src1, descriptive(&register));
        }
        Err(imm) => {
            instr.set_common_field(OperandField::Isimm, boolean(true));
            let value = match imm {
                CmpImm::Constant(value) => int(value),
                CmpImm::Symbol(id) => variable_symbol(id),
            };
            instr.set_common_field(OperandField::Src1, value);
        }
    }
    instr.with_common_comment(&format!("{label}-begin"))
}

/// Replaces: e048_ConstructJADDInstr
///
/// The jump-counter add: a loop counter or a jump counter, plus an immediate.
/// ⭐ THE SAME FOUR ARMS AS [`construct_jsub_instr`], down to `jcr_select` being the only difference
/// between them — see [`JcrOperands`].
#[must_use]
pub fn construct_jadd_instr(operands: JcrOperands, target: RegIndex) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::JADD).with_common_comment("jcr add");
    let (src0, imm) = match operands {
        JcrOperands::Lccr { src0, imm } => (src0, imm),
        JcrOperands::Jcr { src0, imm } => {
            instr.set_common_field(OperandField::JcrSelect, int(1));
            (src0, imm)
        }
    };
    instr.set_common_field(OperandField::JcrTarget, int(i64::from(target.get())));
    instr.set_common_field(OperandField::Src0, int(i64::from(src0.get())));
    instr.set_common_field(OperandField::Imm, int(imm));
    instr
}

/// WHAT A `sentient.add` STEPS AN XRF POINTER BY — the four locale pairs of `:900-935`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XrfAdd {
    /// Which pointer moves.
    pub ptr: XrfPtr,
    /// The step — ⛔ THE OPERAND ORDER IS LOST IN THE REFERENCE TOO: `imm + ptr` and `ptr + imm` both
    /// read the constant into the immediate.
    pub imm: i64,
}

/// Replaces: e049_ConstructXRFADDInstr
///
/// The add that steps one of the PT's XRF pointers, in either operand order.
/// ⭐ ITS EMISSION IS ENTRY 050'S EXACTLY — same opcode, comment, `incr`/`no` pair and wraparound —
/// so it delegates rather than restating it.
#[must_use]
pub fn construct_xrf_add_from_operands<A: Arch>(operands: XrfAdd) -> UniformInstrInfo {
    construct_xrf_add_instr::<A>(operands.ptr, operands.imm)
}

/// WHICH XRF POINTER MOVES — the reference's `is_read` bool, named (`ConstructProgIRHelper.cpp:941`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XrfPtr {
    /// The read pointer — `rdptr_upd:incr`, and the write pointer left alone.
    Read,
    /// The write pointer — `wrptr_upd:incr`.
    Write,
}

/// Replaces: e050_ConstructXRFADDInstr
///
/// Step one of the PT's two XRF pointers by `val`, holding the other still.
///
/// ⛔ THE IMMEDIATE IS FOLDED INTO THE ROW FIRST, so a negative step lands at the top of the row
/// rather than below zero — see [`addr_wraparounded`].
/// ⭐ THE REFERENCE'S `comp` IS UNUSED: only the PT has an XRF.
#[must_use]
pub fn construct_xrf_add_instr<A: Arch>(ptr: XrfPtr, val: i64) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::XRFACCESS).with_common_comment("xrf add");
    let (moved, imm, held) = match ptr {
        XrfPtr::Read => (
            OperandField::RdptrUpd,
            OperandField::RdptrImm,
            OperandField::WrptrUpd,
        ),
        XrfPtr::Write => (
            OperandField::WrptrUpd,
            OperandField::WrptrImm,
            OperandField::RdptrUpd,
        ),
    };
    instr.set_common_field(moved, descriptive("incr"));
    instr.set_common_field(held, descriptive("no"));
    instr.set_common_field(imm, int(addr_wraparounded::<A>(val, AddrSpace::PtXrf)));
    instr
}

/// WHAT A `JADD` ADDS OR A `JSUB` SUBTRACTS — the four locale pairs of `:695-708` and `:973-986`,
/// of which only two outcomes differ, and identically in both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JcrOperands {
    /// An `lccr` and an immediate — a loop counter.
    Lccr {
        /// The register the loop counter lives in.
        src0: RegIndex,
        /// The immediate.
        imm: i64,
    },
    /// A `jcr` and an immediate — ⛔ THE ONLY ARM THAT SETS `jcr_select`.
    Jcr {
        /// The jump-counter register.
        src0: RegIndex,
        /// The immediate.
        imm: i64,
    },
}

/// Replaces: e051_ConstructJSUBInstr
///
/// The jump-counter subtract: a loop counter or a jump counter, less an immediate.
///
/// ⛔ THE REFERENCE DOES NOT DISTINGUISH `a - imm` FROM `imm - a` (`:973-986`) — both orders read the
/// register into `src0` and the constant into `imm`, so the operand order is lost there and here.
#[must_use]
pub fn construct_jsub_instr(operands: JcrOperands, target: RegIndex) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::JSUB).with_common_comment("jcr sub");
    let (src0, imm) = match operands {
        JcrOperands::Lccr { src0, imm } => (src0, imm),
        JcrOperands::Jcr { src0, imm } => {
            instr.set_common_field(OperandField::JcrSelect, int(1));
            (src0, imm)
        }
    };
    instr.set_common_field(OperandField::JcrTarget, int(i64::from(target.get())));
    instr.set_common_field(OperandField::Src0, int(i64::from(src0.get())));
    instr.set_common_field(OperandField::Imm, int(imm));
    instr
}

#[cfg(test)]
mod unit_tests {
    use super::{
        AddrAdd, AddrAddOperands, AddrFile, AddrImm, AddrSub, Assign, AssignKind, Assigned,
        BoundaryTile, CmpImm, CmpOperands, ElseRegion, GtrImm, ImmSource, JcrImm, JcrOperands,
        LoopBound, LrfAdd, LrfAddOperands, LrfImm, LrfSub, LrfSubOperands, ScalarUpdate, SubImm,
        Sync, XrfAdd, XrfPtr, construct_assign_instr, construct_branch_exit_instr,
        construct_jadd_instr, construct_jcmp_instr, construct_jmp_instr, construct_jsub_instr,
        construct_lar_or_ear_add_instr, construct_lar_or_ear_sub_instr, construct_lrf_add_instr,
        construct_lrf_sub_instr, construct_mv_loop_instr, construct_nop_instr,
        construct_return_instr, construct_sync_instr, construct_xrf_add_from_operands,
        construct_xrf_add_instr,
    };
    use crate::arch::{Arch, Dd2, Target};
    use crate::bridges::sentient_to_progir::construct::{boolean, descriptive, instr_tag, int};
    use crate::bridges::sentient_to_progir::state::{
        CopyOps, LabelCounter, Labels, OpSite, RegsToInit,
    };
    use crate::bridges::sentient_to_progir::uniform::instr::Comment;
    use crate::formats::Bits;
    use crate::islands::progir::ty::Operand;
    use crate::islands::progir::{OpCode, OperandField};
    use crate::islands::sentient::dialects::sentient::{
        CmpPredicate, Consumer, Reg, RegIndex, RegType, SyncMode,
    };
    use sys_arch_spec::regfile::Component;

    /// IBM'S OWN `nop.mlir` — three named NOPs and the return
    /// (`dcc/test/Conversion/SentientToProgIR/nop.mlir:8-11`): `SFP_NOP ::  // NOP #1` takes the
    /// dbgName, and an unnamed one would read `// NOP`.
    #[test]
    fn nop_takes_the_dbg_name_or_the_literal() {
        let named = construct_nop_instr(Some("NOP #1"));
        assert_eq!(named.opcode, OpCode::NOP);
        assert_eq!(named.comment, Comment::Common("NOP #1".to_owned()));
        assert_eq!(
            construct_nop_instr(None).comment,
            Comment::Common("NOP".to_owned())
        );
        // ⛔ THE EMPTY STRING SELECTS THE LITERAL rather than surviving as itself.
        assert_eq!(
            construct_nop_instr(Some("")).comment,
            Comment::Common("NOP".to_owned())
        );
    }

    /// `SFP_RETURN ::  // end of the program` (`nop.mlir:12`).
    #[test]
    fn return_carries_ibms_own_comment() {
        let ret = construct_return_instr();
        assert_eq!(ret.opcode, OpCode::RETURN);
        assert_eq!(
            ret.comment,
            Comment::Common("end of the program".to_owned())
        );
    }

    /// IBM'S OWN PT PROGRAM — `PTOP_XRFACCESS :: rdptr_upd:no wrptr_imm:0 wrptr_upd:set` and its read
    /// twin (`test/PT/fp8-bmm.mlir:8-9`). ⭐ THE `set` IS ENTRY 043's SPELLING; this entry writes
    /// `incr`, and what the CHECK line fixes here is WHICH THREE FIELDS each pointer touches.
    #[test]
    fn one_pointer_moves_and_the_other_is_told_not_to() {
        let read = construct_xrf_add_instr::<Dd2>(XrfPtr::Read, 0);
        assert_eq!(read.opcode, OpCode::XRFACCESS);
        assert_eq!(read.comment, Comment::Common("xrf add".to_owned()));
        assert_eq!(
            read.common_fields,
            vec![
                (OperandField::RdptrImm, int(0)),
                (OperandField::RdptrUpd, descriptive("incr")),
                (OperandField::WrptrUpd, descriptive("no")),
            ]
        );
        let write = construct_xrf_add_instr::<Dd2>(XrfPtr::Write, 0);
        assert_eq!(
            write.common_fields,
            vec![
                (OperandField::RdptrUpd, descriptive("no")),
                (OperandField::WrptrImm, int(0)),
                (OperandField::WrptrUpd, descriptive("incr")),
            ]
        );
        // ⛔ A NEGATIVE STEP WRAPS TO THE TOP OF THE ROW rather than staying negative.
        let per_row =
            Target::XRF_CAPACITY.0 / Target::BYTES_PER_STICK.get() / u64::from(Target::PT_ROWS);
        let back = construct_xrf_add_instr::<Target>(XrfPtr::Read, -1);
        assert_eq!(
            back.common_field(OperandField::RdptrImm),
            Some(&int(per_row as i64 - 1))
        );
    }

    /// IBM'S OWN TWO ARMS: `L3_JSUB :: imm:16 jcr_select:1 jcr_target:0 src0:0  // jcr sub`
    /// (`test/Conversion/SentientToProgIR/L3/gather-toggle.mlir:12`) against
    /// `PTOP_JSUB :: imm:2 jcr_target:0 src0:3  // jcr sub` (`test/PT/int8-genkg3-pt.progir:8`) —
    /// ⛔ `jcr_select` IS THE WHOLE DIFFERENCE between a JCR and an LCCR subtract.
    #[test]
    fn only_the_jcr_arm_selects_the_jump_counter() {
        let jcr = construct_jsub_instr(
            JcrOperands::Jcr {
                src0: RegIndex::at::<0>(),
                imm: 16,
            },
            RegIndex::at::<0>(),
        );
        assert_eq!(jcr.opcode, OpCode::JSUB);
        assert_eq!(jcr.comment, Comment::Common("jcr sub".to_owned()));
        assert_eq!(
            jcr.common_fields,
            vec![
                (OperandField::Imm, int(16)),
                (OperandField::JcrSelect, int(1)),
                (OperandField::JcrTarget, int(0)),
                (OperandField::Src0, int(0)),
            ]
        );
        let lccr = construct_jsub_instr(
            JcrOperands::Lccr {
                src0: RegIndex::at::<3>(),
                imm: 2,
            },
            RegIndex::at::<0>(),
        );
        assert_eq!(
            lccr.common_fields,
            vec![
                (OperandField::Imm, int(2)),
                (OperandField::JcrTarget, int(0)),
                (OperandField::Src0, int(3)),
            ]
        );
    }

    /// IBM'S OWN `MVLOOPCNT`s: `PE_MVLOOPCNT :: imm:1792  // for-loop-imm-lccr-0` and
    /// `imm:7  // LR loop #1` (`test/PE/conditional-2and3-nested-if.mlir:8,12`)
    /// against the dynamic `dyn_loop:1 pc_target:(for-loop-jcr-10-end) src0:jcr1
    /// // for-loop-jcr-10-begin` (`if_else_label5.mlir:60`) — ⛔ THE DYNAMIC ARM'S dbgName GOES IN THE
    /// LABEL, the counted arms' in the comment.
    #[test]
    fn a_dynamic_bound_labels_its_own_exit() {
        let mut labels = Labels::default();
        let mut counter = LabelCounter(0);
        let counted = construct_mv_loop_instr(
            &LoopBound::Constant(1792),
            RegIndex::at::<0>(),
            None,
            &mut labels,
            &mut counter,
        );
        assert_eq!(counted.instr.opcode, OpCode::MVLOOPCNT);
        assert_eq!(
            counted.instr.common_fields,
            vec![(OperandField::Imm, int(1792))]
        );
        assert_eq!(
            counted.instr.comment,
            Comment::Common("for-loop-imm-lccr-0".to_owned())
        );
        let named = construct_mv_loop_instr(
            &LoopBound::Constant(7),
            RegIndex::at::<0>(),
            Some("LR loop #1"),
            &mut labels,
            &mut counter,
        );
        assert_eq!(
            named.instr.comment,
            Comment::Common("LR loop #1".to_owned())
        );

        let mut counter = LabelCounter(10);
        let dynamic = construct_mv_loop_instr(
            &LoopBound::Dynamic {
                src0: RegIndex::at::<1>(),
                next: OpSite(3),
            },
            RegIndex::at::<0>(),
            None,
            &mut labels,
            &mut counter,
        );
        assert_eq!(
            dynamic.instr.common_fields,
            vec![
                (OperandField::DynLoop, int(1)),
                (OperandField::PcTarget, instr_tag("for-loop-jcr-10-end")),
                (OperandField::Src0, descriptive("jcr1")),
            ]
        );
        assert_eq!(
            dynamic.instr.comment,
            Comment::Common("for-loop-jcr-10-begin".to_owned())
        );
        // ⛔ THE EXIT LABEL IS LEFT IN THE MAP for whatever lands on that op to carry.
        assert_eq!(labels.get(OpSite(3)), Some("for-loop-jcr-10-end"));
    }

    /// IBM'S OWN SIX ARMS: `PTOP_JIMMCOPY :: imm:1 jcr_target:5  // JCR <- IMM`
    /// (`test/PT/int8-genkg3-pt.progir:18`), `SFP_JADD :: imm:0 jcr_select:1 jcr_target:1 src0:0
    /// // JCR <- JCR` (`if_else_label2.mlir:38`), `LX_IMMCOPY :: lrfimm:-4096 src0:0  // LRF <- IMM`
    /// (`LXLU/int8-kg3-sen1_5-lxlu.mlir:105`), `L3_LARIMM :: imm:0 src0:0  // LAR <- IMM`
    /// (`L3SU/dyn_node_e2e.mlir:45`), `L3_GTRIMM :: imm:513 src0:0
    /// // GTR <- IMM: group_id:2, count:0, num_consumers:1` (`if_else_label.mlir:12`) and
    /// `PTOP_XRFACCESS :: rdptr_upd:no wrptr_imm:0 wrptr_upd:set  // set xrf wr index`
    /// (`PT/fp8-bmm-1p5.mlir:8`) — ⛔ PLUS THE COPY INTO ITS OWN REGISTER, which is no instruction.
    #[test]
    fn each_pair_of_files_has_its_own_opcode_and_comment() {
        let jcr_imm = construct_assign_instr::<Dd2>(
            &Assign {
                src_index: None,
                tgt_index: Some(RegIndex::at::<5>()),
                kind: AssignKind::JcrFromImm(JcrImm::Constant(1)),
            },
            Component::Pt,
            Bits(8),
            false,
        );
        let instr = jcr_imm.instr.expect("an immediate into a JCR always emits");
        assert_eq!(instr.opcode, OpCode::JIMMCOPY);
        assert_eq!(instr.comment, Comment::Common("JCR <- IMM".to_owned()));
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Imm, int(1)),
                (OperandField::JcrTarget, int(5)),
            ]
        );
        assert_eq!(jcr_imm.copy_ops, CopyOps(1));

        let jcr_jcr = construct_assign_instr::<Dd2>(
            &Assign {
                src_index: Some(RegIndex::at::<0>()),
                tgt_index: Some(RegIndex::at::<1>()),
                kind: AssignKind::JcrFromJcr,
            },
            Component::Sfp,
            Bits(8),
            false,
        );
        let instr = jcr_jcr.instr.expect("two different JCRs emit");
        assert_eq!(instr.opcode, OpCode::JADD);
        assert_eq!(instr.comment, Comment::Common("JCR <- JCR".to_owned()));
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Imm, int(0)),
                (OperandField::JcrSelect, int(1)),
                (OperandField::JcrTarget, int(1)),
                (OperandField::Src0, int(0)),
            ]
        );
        // ⭐ A `JADD` OF ZERO IS NOT ONE OF THE SIX COPY OPCODES.
        assert_eq!(jcr_jcr.copy_ops, CopyOps(0));

        let lrf_imm = construct_assign_instr::<Dd2>(
            &Assign {
                src_index: None,
                tgt_index: Some(RegIndex::at::<0>()),
                kind: AssignKind::LrfFromImm(LrfImm::Constant(-4096)),
            },
            Component::Lxlu,
            Bits(8),
            false,
        );
        let instr = lrf_imm.instr.expect("an immediate into an LRF emits");
        assert_eq!(instr.opcode, OpCode::IMMCOPY);
        assert_eq!(instr.comment, Comment::Common("LRF <- IMM".to_owned()));
        // ⛔ `lrfimm` ON AN LX UNIT, where every other component spells the same field `imm`.
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Lrfimm, int(-4096)),
                (OperandField::Src0, int(0)),
            ]
        );
        // ⛔ AND THE PROGRAM HEADER CARRIES IT INSTEAD, emitting nothing.
        assert_eq!(
            construct_assign_instr::<Dd2>(
                &Assign {
                    src_index: None,
                    tgt_index: Some(RegIndex::at::<0>()),
                    kind: AssignKind::LrfFromImm(LrfImm::Constant(-4096)),
                },
                Component::Lxlu,
                Bits(8),
                true,
            )
            .instr,
            None
        );

        let lar_imm = construct_assign_instr::<Dd2>(
            &Assign {
                src_index: None,
                tgt_index: Some(RegIndex::at::<0>()),
                kind: AssignKind::AddrFromImm {
                    file: AddrFile::Lar,
                    imm: AddrImm::Constant(0),
                },
            },
            Component::L3lu,
            Bits(8),
            false,
        );
        let instr = lar_imm.instr.expect("an immediate into a LAR emits");
        assert_eq!(instr.opcode, OpCode::LARIMM);
        assert_eq!(instr.comment, Comment::Common("LAR <- IMM".to_owned()));
        assert_eq!(
            instr.common_fields,
            vec![(OperandField::Imm, int(0)), (OperandField::Src0, int(0)),]
        );

        let gtr = construct_assign_instr::<Dd2>(
            &Assign {
                src_index: None,
                tgt_index: Some(RegIndex::at::<0>()),
                kind: AssignKind::GtrFromImm(GtrImm::Multicast {
                    encoded: 513,
                    decoded: "group_id:2, count:0, num_consumers:1".to_owned(),
                }),
            },
            Component::L3lu,
            Bits(8),
            false,
        );
        let instr = gtr.instr.expect("a multicast group into a GTR emits");
        assert_eq!(instr.opcode, OpCode::GTRIMM);
        // ⛔ THE DECODE REPLACES THE COMMENT, trailing space and all.
        assert_eq!(
            instr.comment,
            Comment::Common("GTR <- IMM: group_id:2, count:0, num_consumers:1".to_owned())
        );
        assert_eq!(
            instr.common_fields,
            vec![(OperandField::Imm, int(513)), (OperandField::Src0, int(0)),]
        );

        let xrf = construct_assign_instr::<Dd2>(
            &Assign {
                src_index: None,
                tgt_index: Some(RegIndex::at::<0>()),
                kind: AssignKind::XrfPtrFromImm {
                    ptr: XrfPtr::Write,
                    val: 0,
                },
            },
            Component::Pt,
            Bits(8),
            false,
        );
        // ⛔ THE POINTER'S OWN INDEX IS WHAT MAKES THIS EMIT AT ALL: `src_index == tgt_index` is
        // tested before the arms, so an unassigned target would be the immediate's own `-1`.
        let instr = xrf.instr.expect("setting an XRF pointer emits");
        assert_eq!(instr.opcode, OpCode::XRFACCESS);
        assert_eq!(
            instr.comment,
            Comment::Common("set xrf wr index".to_owned())
        );
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::RdptrUpd, descriptive("no")),
                (OperandField::WrptrImm, int(0)),
                (OperandField::WrptrUpd, descriptive("set")),
            ]
        );

        // ⛔ A COPY INTO THE REGISTER THE VALUE IS ALREADY IN IS NO INSTRUCTION.
        assert_eq!(
            construct_assign_instr::<Dd2>(
                &Assign {
                    src_index: Some(RegIndex::at::<3>()),
                    tgt_index: Some(RegIndex::at::<3>()),
                    kind: AssignKind::LrfFromLrf,
                },
                Component::Lxlu,
                Bits(8),
                false,
            ),
            Assigned {
                instr: None,
                copy_ops: CopyOps(0),
                refused: Vec::new(),
            }
        );
    }

    /// `SFP_NOP :: be:be  // Branch End` (`test/SFP/merge_and_pack.mlir:111`).
    #[test]
    fn a_branch_ends_on_a_nop_that_says_so() {
        let exit = construct_branch_exit_instr();
        assert_eq!(exit.opcode, OpCode::NOP);
        assert_eq!(exit.comment, Comment::Common("Branch End".to_owned()));
        assert_eq!(
            exit.common_fields,
            vec![(OperandField::Be, descriptive("be"))]
        );
    }

    /// IBM'S OWN THREE: `LX_SYNC :: synctag:167  // sync sendrecv-lxsu-lxsuN-l3lu`
    /// (`LXLU/sync-op-lxlu-l3.mlir:27`), `L3_SYNC :: soft:yes synctag:81
    /// // sync send-lxlu0-lxlu1-soft` (`L3/reginit_uniform.mlir:13`) and
    /// `L0_SYNC :: implicit:yes tilesize:32  // sync #1` (`implicit-sync.mlir:9`) — ⛔ THE THIRD LINE
    /// HAS NO `synctag` AT ALL, and the same four bits mean LX units on L3 and L3 units on LX.
    #[test]
    fn the_peers_are_bits_in_the_synctag_unless_the_sync_is_implicit() {
        let lx = construct_sync_instr::<Dd2>(
            &Sync::Lx {
                mode: SyncMode::SendRecv,
                peers: vec![Consumer::Lxsu, Consumer::LxsuN, Consumer::L3lu],
            },
            false,
            None,
        );
        assert_eq!(lx.opcode, OpCode::SYNC);
        assert_eq!(lx.common_fields, vec![(OperandField::Synctag, int(167))]);
        assert_eq!(
            lx.comment,
            Comment::Common("sync sendrecv-lxsu-lxsuN-l3lu".to_owned())
        );

        let l3 = construct_sync_instr::<Dd2>(
            &Sync::L3lu {
                mode: SyncMode::Send,
                peers: vec![Consumer::Lxlu0, Consumer::Lxlu1],
                soft: true,
            },
            false,
            None,
        );
        assert_eq!(
            l3.common_fields,
            vec![
                (OperandField::Soft, descriptive("yes")),
                (OperandField::Synctag, int(81)),
            ]
        );
        assert_eq!(
            l3.comment,
            Comment::Common("sync send-lxlu0-lxlu1-soft".to_owned())
        );

        let implicit = construct_sync_instr::<Dd2>(
            &Sync::L0Implicit {
                tile: BoundaryTile(32),
            },
            false,
            Some("sync #1"),
        );
        assert_eq!(
            implicit.common_fields,
            vec![
                (OperandField::Implicit, descriptive("yes")),
                (OperandField::Tilesize, int(32)),
            ]
        );
        assert_eq!(implicit.comment, Comment::Common("sync #1".to_owned()));
    }

    /// `L3_JCMP :: mode:always pc_target:(if-label-0-end)  // jump` (`L3SU/dyn_node_e2e.mlir:49`).
    #[test]
    fn an_unconditional_jump_is_a_compare_that_always_holds() {
        let jmp = construct_jmp_instr("if-label-0-end");
        assert_eq!(jmp.opcode, OpCode::JCMP);
        assert_eq!(jmp.comment, Comment::Common("jump".to_owned()));
        assert_eq!(
            jmp.common_fields,
            vec![
                (OperandField::Mode, descriptive("always")),
                (OperandField::PcTarget, instr_tag("if-label-0-end")),
            ]
        );
    }

    /// `L3_JCMP :: isimm:true mode:ne pc_target:(if-label-0-else) src0:LCCR1 src1:2 usejcr:false
    /// // if-label-0-begin` (`L3SU/dyn_node_e2e.mlir:47`) — ⛔ `eq` BECOMES `ne`: the jump is taken
    /// when the condition FAILS, and it goes to the `else` body when there is one.
    #[test]
    fn the_emitted_predicate_is_the_inverse_of_the_compare() {
        let mut labels = Labels::default();
        let mut counter = LabelCounter(0);
        let jcmp = construct_jcmp_instr(
            CmpPredicate::Eq,
            CmpOperands::LccrVsImm {
                lccr: RegIndex::at::<1>(),
                imm: CmpImm::Constant(2),
            },
            &ElseRegion::Head {
                at: OpSite(7),
                yields: None,
            },
            &[],
            OpSite(9),
            &mut labels,
            &mut counter,
        );
        assert_eq!(jcmp.opcode, OpCode::JCMP);
        assert_eq!(
            jcmp.common_fields,
            vec![
                (OperandField::Isimm, boolean(true)),
                (OperandField::Mode, descriptive("ne")),
                (OperandField::PcTarget, instr_tag("if-label-0-else")),
                (OperandField::Src0, descriptive("LCCR1")),
                (OperandField::Src1, int(2)),
                (OperandField::Usejcr, boolean(false)),
            ]
        );
        assert_eq!(jcmp.comment, Comment::Common("if-label-0-begin".to_owned()));
        // ⛔ BOTH LABELS ARE CLAIMED even though only the `else` one is jumped to.
        assert_eq!(labels.get(OpSite(7)), Some("if-label-0-else"));
        assert_eq!(labels.get(OpSite(9)), Some("if-label-0-end"));

        // ⛔ AN `else` THAT ONLY YIELDS THE `if`'s OWN RESULTS IS JUMPED PAST: the end label, not the
        // else label, and `sgt` inverts to `ge` only because the immediate is on the RIGHT.
        let mut counter = LabelCounter(1);
        let held = Reg {
            locale: RegType::Lccr,
            index: Some(RegIndex::at::<1>()),
        };
        let past = construct_jcmp_instr(
            CmpPredicate::Sgt,
            CmpOperands::LccrVsImm {
                lccr: RegIndex::at::<1>(),
                imm: CmpImm::Constant(2),
            },
            &ElseRegion::Head {
                at: OpSite(7),
                yields: Some(vec![held]),
            },
            &[held],
            OpSite(9),
            &mut labels,
            &mut counter,
        );
        assert_eq!(
            past.common_field(OperandField::PcTarget),
            Some(&instr_tag("if-label-1-end"))
        );
        assert_eq!(
            past.common_field(OperandField::Mode),
            Some(&descriptive("ge"))
        );
    }

    /// `PTOP_JADD :: imm:-2 jcr_select:1 jcr_target:2 src0:0  // jcr add`
    /// (`test/PT/int8-genkg3-pt.progir:14`) — ⭐ THE SAME SHAPE AS THE `JSUB` ABOVE, `jcr_select` and
    /// the opcode apart.
    #[test]
    fn the_add_selects_the_jump_counter_on_the_same_arm_the_sub_does() {
        let jadd = construct_jadd_instr(
            JcrOperands::Jcr {
                src0: RegIndex::at::<0>(),
                imm: -2,
            },
            RegIndex::at::<2>(),
        );
        assert_eq!(jadd.opcode, OpCode::JADD);
        assert_eq!(jadd.comment, Comment::Common("jcr add".to_owned()));
        assert_eq!(
            jadd.common_fields,
            vec![
                (OperandField::Imm, int(-2)),
                (OperandField::JcrSelect, int(1)),
                (OperandField::JcrTarget, int(2)),
                (OperandField::Src0, int(0)),
            ]
        );
        let lccr = construct_jadd_instr(
            JcrOperands::Lccr {
                src0: RegIndex::at::<0>(),
                imm: -2,
            },
            RegIndex::at::<2>(),
        );
        assert_eq!(lccr.common_field(OperandField::JcrSelect), None);
    }

    /// `PTOP_XRFACCESS :: rdptr_upd:no wrptr_imm:16 wrptr_upd:incr  // xrf add`
    /// (`if_else_label4.mlir:16`) — ⭐ BYTE-FOR-BYTE ENTRY 050'S INSTRUCTION, which is why this entry
    /// delegates to it.
    #[test]
    fn the_operand_form_emits_what_entry_050_does() {
        let add = construct_xrf_add_from_operands::<Dd2>(XrfAdd {
            ptr: XrfPtr::Write,
            imm: 16,
        });
        assert_eq!(add.opcode, OpCode::XRFACCESS);
        assert_eq!(add.comment, Comment::Common("xrf add".to_owned()));
        assert_eq!(
            add.common_fields,
            vec![
                (OperandField::RdptrUpd, descriptive("no")),
                (OperandField::WrptrImm, int(16)),
                (OperandField::WrptrUpd, descriptive("incr")),
            ]
        );
        assert_eq!(add, construct_xrf_add_instr::<Dd2>(XrfPtr::Write, 16));
    }

    /// Each instruction as the dump prints it: opcode, fields in ISA order, comment.
    fn emitted(out: &ScalarUpdate) -> Vec<(OpCode, Vec<(OperandField, Operand)>, String)> {
        out.instrs
            .iter()
            .map(|instr| {
                let comment = match &instr.comment {
                    Comment::Common(text) => text.clone(),
                    _ => String::new(),
                };
                (instr.opcode, instr.common_fields.clone(), comment)
            })
            .collect()
    }

    /// ⭐⭐ IBM'S OWN PAIR — `LX_LRFREGCOPY :: src0:3 src1:1  // LRF <- LRF` then
    /// `LX_MODLRFREG :: src0:3 src1:0  // lrf add`
    /// (`lx_indirect_loads_stores_composite.mlir:15-16`): a target that is neither operand.
    #[test]
    fn an_lrf_add_copies_when_the_target_is_neither_operand() {
        let out = construct_lrf_add_instr::<Dd2>(
            &LrfAdd {
                tgt: Some(RegIndex::at::<3>()),
                operands: LrfAddOperands::RegReg {
                    src0: Some(RegIndex::at::<1>()),
                    src1: Some(RegIndex::at::<0>()),
                },
            },
            Component::Lxlu,
            Bits(8),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(out.copy_ops, CopyOps(1));
        assert_eq!(out.refused, vec![]);
        assert_eq!(
            emitted(&out),
            vec![
                (
                    OpCode::LRFREGCOPY,
                    vec![(OperandField::Src0, int(3)), (OperandField::Src1, int(1))],
                    "LRF <- LRF".to_owned()
                ),
                (
                    OpCode::MODLRFREG,
                    vec![(OperandField::Src0, int(3)), (OperandField::Src1, int(0))],
                    "lrf add".to_owned()
                ),
            ]
        );
    }

    /// ⭐⭐ IBM'S OWN LINES — `L3_MODLARREG :: src0:1 src1:0  // lar add` and
    /// `L3_ADDLARIMM :: imm:10 src0:0  // lar add` (`conditional-addr.mlir:17,19`). ⛔ THE L3'S
    /// ADDRESSES COUNT IN 128-BYTE UNITS, so `1280` bits of immediate is an `imm:10`.
    #[test]
    fn a_lar_add_scales_its_immediate_and_takes_the_other_operand() {
        let of = |operands| AddrAdd {
            file: AddrFile::Lar,
            tgt: Some(RegIndex::at::<1>()),
            operands,
        };
        let reg_reg = construct_lar_or_ear_add_instr::<Dd2>(
            &of(AddrAddOperands::RegReg {
                src0: Some(RegIndex::at::<1>()),
                src1: Some(RegIndex::at::<0>()),
            }),
            Component::L3lu,
            Bits(8),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(reg_reg.copy_ops, CopyOps(0));
        assert_eq!(
            emitted(&reg_reg),
            vec![(
                OpCode::MODLARREG,
                vec![(OperandField::Src0, int(1)), (OperandField::Src1, int(0))],
                "lar add".to_owned()
            )]
        );
        let reg_imm = construct_lar_or_ear_add_instr::<Dd2>(
            &AddrAdd {
                file: AddrFile::Lar,
                tgt: Some(RegIndex::at::<0>()),
                operands: AddrAddOperands::RegImm {
                    reg: Some(RegIndex::at::<0>()),
                    imm: ImmSource::Constant(1280),
                },
            },
            Component::L3lu,
            Bits(8),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(
            emitted(&reg_imm),
            vec![(
                OpCode::ADDLARIMM,
                vec![(OperandField::Imm, int(10)), (OperandField::Src0, int(0))],
                "lar add".to_owned()
            )]
        );
    }

    /// ⭐⭐ IBM'S OWN PAIR — `LX_LRFREGCOPY :: src0:0 src1:2  // LRF <- LRF` then
    /// `LX_SUBLRFIMM :: lrfimm:12032 src0:0  // lrf sub` (`unordered_constant_map.mlir:44-45`), and
    /// ⛔ THE 2 MB SPLIT: an immediate over `0x1FFFFF` leaves the remainder to a following modify.
    #[test]
    fn an_lrf_sub_copies_first_and_splits_an_immediate_over_two_megabytes() {
        let of = |tgt, imm| LrfSub {
            tgt,
            operands: LrfSubOperands::ImmReg {
                imm,
                src1: Some(RegIndex::at::<2>()),
            },
        };
        let out = construct_lrf_sub_instr::<Dd2>(
            &of(Some(RegIndex::at::<0>()), SubImm::Constant(12032)),
            Component::Lxlu,
            Bits(8),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(out.copy_ops, CopyOps(1));
        assert_eq!(
            emitted(&out),
            vec![
                (
                    OpCode::LRFREGCOPY,
                    vec![(OperandField::Src0, int(0)), (OperandField::Src1, int(2))],
                    "LRF <- LRF".to_owned()
                ),
                (
                    OpCode::SUBLRFIMM,
                    vec![
                        (OperandField::Lrfimm, int(12032)),
                        (OperandField::Src0, int(0))
                    ],
                    "lrf sub".to_owned()
                ),
            ]
        );
        let split = construct_lrf_sub_instr::<Dd2>(
            &of(Some(RegIndex::at::<2>()), SubImm::Constant(0x20_0000)),
            Component::Lxlu,
            Bits(8),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(split.copy_ops, CopyOps(0));
        assert_eq!(
            emitted(&split),
            vec![
                (
                    OpCode::SUBLRFIMM,
                    vec![
                        (OperandField::Lrfimm, int(0x1F_FFFF)),
                        (OperandField::Src0, int(2))
                    ],
                    "lrf sub".to_owned()
                ),
                // ⭐ THE REMAINDER'S MODIFY CARRIES NO COMMENT OF ITS OWN.
                (
                    OpCode::MODLRFIMM,
                    vec![(OperandField::Lrfimm, int(1)), (OperandField::Src0, int(2))],
                    String::new()
                ),
            ]
        );
    }

    /// ⭐⭐ IBM'S OWN LINE — `L3_SUBLARIMM :: imm:1824 src0:0  // lar sub`
    /// (`uniform_scalar_sub.mlir:9`), the source already in the target register.
    #[test]
    fn a_lar_sub_needs_no_copy_when_the_source_is_the_target() {
        let out = construct_lar_or_ear_sub_instr::<Dd2>(
            &AddrSub {
                tgt: Some(RegIndex::at::<0>()),
                imm: ImmSource::Constant(233_472),
                src1: Some(RegIndex::at::<0>()),
            },
            Component::L3lu,
            Bits(8),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(out.copy_ops, CopyOps(0));
        assert_eq!(
            emitted(&out),
            vec![(
                OpCode::SUBLARIMM,
                vec![(OperandField::Imm, int(1824)), (OperandField::Src0, int(0))],
                "lar sub".to_owned()
            )]
        );
    }
}
