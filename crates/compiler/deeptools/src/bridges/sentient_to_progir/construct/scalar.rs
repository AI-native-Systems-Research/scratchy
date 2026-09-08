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

use crate::arch::Arch;
use crate::bridges::sentient_to_progir::construct::{descriptive, int};
use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::bridges::sentient_to_progir::utils::{AddrSpace, addr_wraparounded};
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::RegIndex;

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

// crustify:todo: e042_ConstructMVLoopInstr
// crustify:todo: e043_ConstructAssignInstr
// crustify:todo: e044_ConstructBranchExitInstr
// crustify:todo: e045_ConstructSyncInstr
// crustify:todo: e046_ConstructJMPInstr
// crustify:todo: e047_ConstructJCMPInstr
// crustify:todo: e048_ConstructJADDInstr
// crustify:todo: e049_ConstructXRFADDInstr
// crustify:todo: e081_ConstructLRFADDInstr
// crustify:todo: e082_ConstructLARorEARADDInstr
// crustify:todo: e083_ConstructLRFSUBInstr
// crustify:todo: e084_ConstructLARorEARSUBInstr

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

/// WHAT A `JSUB` SUBTRACTS — the four locale pairs of `:971-985`, of which only two outcomes differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JcrSubOperands {
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
pub fn construct_jsub_instr(operands: JcrSubOperands, target: RegIndex) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::JSUB).with_common_comment("jcr sub");
    let (src0, imm) = match operands {
        JcrSubOperands::Lccr { src0, imm } => (src0, imm),
        JcrSubOperands::Jcr { src0, imm } => {
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
        JcrSubOperands, XrfPtr, construct_jsub_instr, construct_nop_instr, construct_return_instr,
        construct_xrf_add_instr,
    };
    use crate::arch::{Arch, Dd2, Target};
    use crate::bridges::sentient_to_progir::construct::{descriptive, int};
    use crate::bridges::sentient_to_progir::uniform::instr::Comment;
    use crate::islands::progir::{OpCode, OperandField};
    use crate::islands::sentient::dialects::sentient::RegIndex;

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
    /// twin (`test/PT/fp8-bmm.mlir:8-9`). ⭐ THE `set` IS ENTRY 049's SPELLING; this entry writes
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
            JcrSubOperands::Jcr {
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
            JcrSubOperands::Lccr {
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
}
