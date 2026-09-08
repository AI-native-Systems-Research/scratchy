//! THE MASK, SPLAT AND SAMV INSTRUCTIONS — set-dest-mask, set-dest, splat, splat-pad, and the
//! set/incr mask pair.
//!
//! 9 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e005_ConstructIncrMaskInstr` | 0 | 11 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4118` |
//! | `e060_ConstructSetDstMaskInstr` | 1 | 41 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3559` |
//! | `e061_ConstructSetDestInstr` | 1 | 54 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3601` |
//! | `e062_ConstructImmCopyInstrFromSplatOp` | 1 | 60 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3657` |
//! | `e064_ConstructSAMVInstr` | 1 | 33 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4034` |
//! | `e065_ConstructSAMVResetInstruction` | 1 | 26 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4068` |
//! | `e066_ConstructSetMaskInstr` | 1 | 22 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4095` |
//! | `e092_ConstructSplatInstrFromSplatOp` | 2 | 65 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3719` |
//! | `e098_ConstructSplatPadInstr` | 3 | 65 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3792` |

use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::islands::progir::OpCode;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e005_ConstructIncrMaskInstr
///
/// Advance the PT's mask by one.
///
/// ⛔ PT ONLY — the reference aborts on any other unit (`:4121-4122`). There is no component
/// parameter here, so a call for the SFP or a load unit cannot be written rather than being refused.
///
/// ⭐ AN UNNAMED OP GETS NO COMMENT AT ALL, not an empty one: the reference only calls
/// `setCommonComment` when `dbgName` is present, unlike [`construct_nop_instr`]'s literal fallback.
#[must_use]
pub fn construct_incr_mask_instr(dbg_name: Option<&str>) -> UniformInstrInfo {
    let instr = UniformInstrInfo::of(OpCode::INCRMASK);
    match dbg_name {
        Some(name) => instr.with_common_comment(name),
        None => instr,
    }
}

// crustify:todo: e060_ConstructSetDstMaskInstr
// crustify:todo: e061_ConstructSetDestInstr
// crustify:todo: e062_ConstructImmCopyInstrFromSplatOp
// crustify:todo: e064_ConstructSAMVInstr
// crustify:todo: e065_ConstructSAMVResetInstruction
// crustify:todo: e066_ConstructSetMaskInstr
// crustify:todo: e092_ConstructSplatInstrFromSplatOp
// crustify:todo: e098_ConstructSplatPadInstr

#[cfg(test)]
mod unit_tests {
    use super::construct_incr_mask_instr;
    use crate::bridges::sentient_to_progir::uniform::instr::Comment;
    use crate::islands::progir::OpCode;

    /// IBM'S OWN `setmask_incrmask.mlir:13` — `PTOP_INCRMASK ::  // incr_mask #1`: the opcode, the
    /// dbgName as the comment, and NO operand fields.
    #[test]
    fn incr_mask_carries_only_the_dbg_name() {
        let named = construct_incr_mask_instr(Some("incr_mask #1"));
        assert_eq!(named.opcode, OpCode::INCRMASK);
        assert_eq!(named.comment, Comment::Common("incr_mask #1".to_owned()));
        assert!(named.common_fields.is_empty());
        assert_eq!(construct_incr_mask_instr(None).comment, Comment::None);
    }
}
