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

use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::islands::progir::OpCode;

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
// crustify:todo: e050_ConstructXRFADDInstr
// crustify:todo: e051_ConstructJSUBInstr
// crustify:todo: e081_ConstructLRFADDInstr
// crustify:todo: e082_ConstructLARorEARADDInstr
// crustify:todo: e083_ConstructLRFSUBInstr
// crustify:todo: e084_ConstructLARorEARSUBInstr

#[cfg(test)]
mod unit_tests {
    use super::{construct_nop_instr, construct_return_instr};
    use crate::bridges::sentient_to_progir::uniform::instr::Comment;
    use crate::islands::progir::OpCode;

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
}
