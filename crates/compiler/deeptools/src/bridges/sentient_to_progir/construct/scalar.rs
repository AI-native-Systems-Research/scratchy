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
use crate::bridges::sentient_to_progir::construct::{
    boolean, descriptive, instr_tag, int, variable_symbol,
};
use crate::bridges::sentient_to_progir::lower::labels_and_regs::address_scale;
use crate::bridges::sentient_to_progir::state::{CopyOps, LabelCounter, Labels, OpSite};
use crate::bridges::sentient_to_progir::uniform::instr::{
    MapMode, MappedEntry, OperandMapRefusal, UniformInstrInfo, add_entry_to_operand_map,
};
use crate::bridges::sentient_to_progir::utils::{AddrSpace, addr_wraparounded};
use crate::formats::Bits;
use crate::islands::progir::ty::Operand;
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{
    CmpPredicate, Consumer, Reg, RegIndex, RegType, SyncMode,
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

// crustify:todo: e081_ConstructLRFADDInstr
// crustify:todo: e082_ConstructLARorEARADDInstr
// crustify:todo: e083_ConstructLRFSUBInstr
// crustify:todo: e084_ConstructLARorEARSUBInstr

/// `getValueRegIndex`'s answer as a field value — ⛔ THE REFERENCE WRITES ITS `-1` STRAIGHT INTO THE
/// FIELD (`SentientOps.cpp:1886`), so an unassigned register is a negative index in the program.
fn index_field(index: Option<RegIndex>) -> Operand {
    int(index.map_or(-1, |at| i64::from(at.get())))
}

/// `setOperandMap(OperandMap(…))` — ⛔ IT REPLACES THE MAP (`UniformInstrAndBlock.hpp:132`): every
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

/// WHAT A `sentient.for`'s TRIP COUNT IS — the four defining ops of `:57-107`.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopBound {
    /// `sentient.constant` — one trip count for every unit.
    Constant(i64),
    /// `uniform.query_map` — one per unit; ⛔ MODE `none` AND SCALE 1, `OperandMap`'s own defaults
    /// (`UniformInstrAndBlock.hpp:31-34`), which this call site leaves unstated.
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
/// ⛔ THE DYNAMIC ARM READS THE LABEL MAP BACK (`:104-107`) — a loop whose successor is already
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

/// WHAT A `JIMMCOPY`'s IMMEDIATE IS (`:142-151`).
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

/// WHAT A `LARIMM`/`EARIMM`'s IMMEDIATE IS (`:243-288`).
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

/// WHAT A `GTRIMM`'s IMMEDIATE IS (`:303-314`).
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

/// WHICH TWO FILES AN ASSIGNMENT COPIES BETWEEN — the arms of `:131-315`.
///
/// ⛔ `LBR`/`EBR <- IMM` IS UNREPRESENTABLE, which is the reference's own `DT_CHECK` (`:128-130`).
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
    /// `++num_copy_ops_` — 1 for each of the six copy opcodes (`:317-321`).
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
/// ⛔ A COPY INTO THE REGISTER THE VALUE IS ALREADY IN IS NO INSTRUCTION (`:126`), which also makes
/// the reference's same-locale arm (`:229-231`) dead code.
/// ⛔ `program_header` DIVERTS ONLY THE TWO IMMEDIATE ARMS (`:159,232`).
/// ⛔ AND THE LAR/EAR IMMEDIATE WRITES `imm` WHERE ITS MAPPED TWIN WRITES `lrfimm` (`:246,264`).
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
    let scale = |locale: RegType| {
        f64::from(element_size.0) / 8.0 / f64::from(address_scale::<A>(comp, locale).get())
    };
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
                    let imm = (*value as f64 * scale(RegType::Lrf)) as i64;
                    instr.set_common_field(imm_field, int(imm));
                }
                LrfImm::Mapped(entries) => {
                    refused = map_into(
                        &mut instr,
                        imm_field,
                        entries,
                        MapMode::None,
                        scale(RegType::Lrf),
                    );
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
            let (opcode, comment, locale) = match file {
                AddrFile::Lar => (OpCode::LARIMM, "LAR <- IMM", RegType::Lar),
                AddrFile::Ear => (OpCode::EARIMM, "EAR <- IMM", RegType::Ear),
            };
            let mut instr = UniformInstrInfo::of(opcode).with_common_comment(comment);
            instr.set_common_field(OperandField::Src0, index_field(assign.tgt_index));
            match imm {
                AddrImm::Constant(value) => {
                    let imm = (*value as f64 * scale(locale)) as i64;
                    instr.set_common_field(OperandField::Imm, int(imm));
                }
                AddrImm::Core(core) => {
                    instr.set_common_field(OperandField::Imm, int(i64::from(core.get())));
                }
                AddrImm::MappedConstants(entries) => {
                    refused =
                        map_into(&mut instr, imm_field, entries, MapMode::None, scale(locale));
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
            // ⛔ THE TRAILING SPACE IS THE REFERENCE'S (`:302`), and only the mapped arm keeps it.
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
    // `:317-321` — ⭐ `LRFCOPY` IS IN THE SET AND NO ARM HERE EMITS IT.
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

/// `lx_consumer_map` (`:355-357`) — ⛔ A CONSUMER THE MAP DOES NOT NAME CONTRIBUTES 0, which is what
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

/// `l3_consumer_map` (`:358-360`) — ⭐ THE SAME FOUR BITS NAME THE PER-CORELET LX UNITS HERE.
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
    /// An `L3LU` sync — [`l3_bit`], and ⛔ THE ONLY COMPONENT A SOFT SYNC IS ALLOWED ON (`:391`).
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
    /// CONSTRUCTION (`:404`) and ⛔ IT WRITES NO `synctag` AT ALL (`:421`).
    L0Implicit {
        /// `implicit_sync_memory_boundary`.
        tile: BoundaryTile,
    },
}

/// Replaces: e045_ConstructSyncInstr
///
/// The rendezvous: a mode and one bit per peer in `synctag`, or the L0's implicit tile boundary.
///
/// ⛔ AN IMPLICIT SYNC RETURNS BEFORE `synctag` IS SET (`:421`) — its tag is the tile, not the peers.
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
    // `:412-419` — ⭐ SEN1P5 NEEDS A DESTINATION ON EVERY L0 SYNC, implicit or not.
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

/// WHAT A `sentient.if` COMPARES ITS REGISTER AGAINST (`:610-625,663-677`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpImm {
    /// `sentient.constant` — the value, into `src1`.
    Constant(i64),
    /// `symbol.create_symbol` — ⛔ ITS ID INTO `src1` AS A `VARIABLE_SYMBOL`, and the id's absence is
    /// unrepresentable where the reference reports it (`:667-670`).
    Symbol(i64),
}

/// WHICH REGISTERS A `sentient.if` COMPARES — the seven operand arms of `:625-660`.
///
/// ⛔ WHICH SIDE THE IMMEDIATE IS ON IS LOAD-BEARING: it decides the inverted mode, and only an
/// immediate on the RIGHT changes it — ⭐ the reference's `lhs == imm` and neither-is-imm branches
/// agree in all four signed predicates (`:487-544`).
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
    /// INSTRUCTION (`:645-653`): `src0` is the LCCR and `src1` the JCR whichever side each was on,
    /// and with no immediate the mode does not move either.
    LccrAndJcr {
        /// `src0`.
        lccr: RegIndex,
        /// `src1`.
        jcr: RegIndex,
    },
    /// `jcr CMP jcr` — ⛔ `src0` IS THE RIGHT-HAND ONE (`:655-659`).
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
    /// No `else` region at all (`:600`).
    None,
    /// An `else` region and its first op.
    Head {
        /// The op the `-else` label would be claimed for.
        at: OpSite,
        /// ⛔ `Some` ONLY WHERE THAT FIRST OP IS A `sentient.yield` (`:571`), carrying the registers it
        /// yields: an else that only yields what the `if` already holds needs no label at all.
        yields: Option<Vec<Reg>>,
    },
}

/// Replaces: e047_ConstructJCMPInstr
///
/// The conditional jump a `sentient.if` becomes: the INVERTED predicate, because the branch is taken
/// when the condition fails, and the label of the `else` body or of the op after the `if`.
///
/// ⛔ THE END LABEL IS NOT READ BACK (`:559-562`), unlike [`construct_mv_loop_instr`]'s: a successor
/// that already carries a label keeps it, and `pc_target` still names the one minted here.
/// ⛔ AN `else` THAT YIELDS EXACTLY THE `if`'s OWN RESULTS IS JUMPED PAST, not into (`:568-598`).
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
    // `:487-544` — the inversion, which the immediate's side flips back for.
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
            // `:571-596` — a yield of the `if`'s own results generates no instructions to jump to.
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

/// WHAT A `JADD` ADDS OR A `JSUB` SUBTRACTS — the four locale pairs of `:693-708` and `:971-985`,
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
/// ⛔ THE REFERENCE DOES NOT DISTINGUISH `a - imm` FROM `imm - a` (`:971-985`) — both orders read the
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
        AddrFile, AddrImm, Assign, AssignKind, Assigned, BoundaryTile, CmpImm, CmpOperands,
        ElseRegion, GtrImm, JcrImm, JcrOperands, LoopBound, LrfImm, Sync, XrfAdd, XrfPtr,
        construct_assign_instr, construct_branch_exit_instr, construct_jadd_instr,
        construct_jcmp_instr, construct_jmp_instr, construct_jsub_instr, construct_mv_loop_instr,
        construct_nop_instr, construct_return_instr, construct_sync_instr,
        construct_xrf_add_from_operands, construct_xrf_add_instr,
    };
    use crate::arch::{Arch, Dd2, Target};
    use crate::bridges::sentient_to_progir::construct::{boolean, descriptive, instr_tag, int};
    use crate::bridges::sentient_to_progir::state::{CopyOps, LabelCounter, Labels, OpSite};
    use crate::bridges::sentient_to_progir::uniform::instr::Comment;
    use crate::formats::Bits;
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
    /// `imm:7  // LR loop #1` (`test/Conversion/SentientToProgIR/PE/conditional-2and3-nested-if.mlir:8,12`)
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

    /// `SFP_NOP :: be:be  // Branch End` (`test/Conversion/SentientToProgIR/SFP/merge_and_pack.mlir:111`).
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
}
