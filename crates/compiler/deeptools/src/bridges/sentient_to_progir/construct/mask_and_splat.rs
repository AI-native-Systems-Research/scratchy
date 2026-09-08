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

use super::{descriptive, int};
use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{RawPrecision, RegIndex};

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
/// Replaces: e065_ConstructSAMVResetInstruction
///
/// Put the mask register back: one slice, one bit wide, no valid entries.
/// ⛔ LXLU ONLY, SO THERE IS NO COMPONENT PARAMETER (`:4071-4072`).
/// ⛔ AND THE PROGRAM'S OWN SAMV IS AN INPUT, NOT A FLAG: the mask register and the precision are the
/// two fields this reset does not hardcode, so they arrive as values and
/// `DT_CHECK_MSG(has_samv_, "No SAMV was detected in the program")` has nothing left to refuse.
#[must_use]
pub fn construct_samv_reset_instruction(
    mask_value: RegIndex,
    precision: RawPrecision,
) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::SAMV);
    instr.set_common_field(OperandField::Maskall, descriptive("no"));
    instr.set_common_field(OperandField::Sliceidxsl, int(7));
    instr.set_common_field(OperandField::Numvalidentry, int(0));
    instr.set_common_field(
        OperandField::Mvridx,
        descriptive(&format!("MVR{}", mask_value.get())),
    );
    instr.set_common_field(
        OperandField::Precision,
        descriptive(&format!("{}b", precision.0)),
    );
    instr.set_common_field(OperandField::Xslinner, descriptive("no"));
    instr.set_common_field(OperandField::Wsllen, descriptive("1b"));
    instr
}

/// Replaces: e066_ConstructSetMaskInstr
///
/// Load the PT's mask register with a constant.
/// ⛔ PT ONLY, SO THERE IS NO COMPONENT PARAMETER (`:4097-4098`).
/// ⛔ AND THE MASK VALUE IS A CONSTANT BY THE TYPE: `llvm_unreachable("expecting ConstantOp from
/// mask_value operand")` (`:4111`) is unreachable when a non-constant cannot be written down.
#[must_use]
pub fn construct_set_mask_instr(mask_value: i64, dbg_name: Option<&str>) -> UniformInstrInfo {
    let instr = UniformInstrInfo::of(OpCode::SETMASK);
    let mut instr = match dbg_name {
        Some(name) => instr.with_common_comment(name),
        None => instr,
    };
    instr.set_common_field(OperandField::Imm, int(mask_value));
    instr
}

// crustify:todo: e092_ConstructSplatInstrFromSplatOp
// crustify:todo: e098_ConstructSplatPadInstr

#[cfg(test)]
mod unit_tests {
    use super::{
        OperandField, RawPrecision, RegIndex, construct_incr_mask_instr,
        construct_samv_reset_instruction, construct_set_mask_instr, descriptive, int,
    };
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

    /// IBM'S OWN `samv.mlir:11` — `LX_SAMV :: maskall:no mvridx:MVR0 numvalidentry:0 precision:8b
    /// sliceidxsl:7 wsllen:1b xslinner:no  // samv #4`, which is the reset's whole field set.
    #[test]
    fn the_samv_reset_hardcodes_every_field_but_the_mask_register_and_the_precision() {
        let reset = construct_samv_reset_instruction(RegIndex::at::<0>(), RawPrecision(8));
        assert_eq!(reset.opcode, OpCode::SAMV);
        assert_eq!(
            reset.common_fields,
            vec![
                (OperandField::Maskall, descriptive("no")),
                (OperandField::Mvridx, descriptive("MVR0")),
                (OperandField::Numvalidentry, int(0)),
                (OperandField::Precision, descriptive("8b")),
                (OperandField::Sliceidxsl, int(7)),
                (OperandField::Wsllen, descriptive("1b")),
                (OperandField::Xslinner, descriptive("no")),
            ]
        );
        assert_eq!(reset.comment, Comment::None);
    }

    /// IBM'S OWN `setmask_incrmask.mlir:11,17` — `PTOP_SETMASK :: imm:0  // set_mask #1` and
    /// `imm:2  // set_mask #2`.
    #[test]
    fn set_mask_carries_the_constant_as_its_immediate() {
        let first = construct_set_mask_instr(0, Some("set_mask #1"));
        assert_eq!(first.opcode, OpCode::SETMASK);
        assert_eq!(first.common_fields, vec![(OperandField::Imm, int(0))]);
        assert_eq!(first.comment, Comment::Common("set_mask #1".to_owned()));
        let second = construct_set_mask_instr(2, None);
        assert_eq!(second.common_fields, vec![(OperandField::Imm, int(2))]);
        assert_eq!(second.comment, Comment::None);
    }
}
